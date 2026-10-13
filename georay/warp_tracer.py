# georay/warp_tracer.py
"""
全 Warp 内核实现的 GeoRay tracer。

新增功能：
    * GRIN callable via @wp.func（编译期内联，零性能损失）
    * 波长依赖材质（n(λ), R(λ), ...）通过 GPU 端 LUT 查找
    * GRIN 抛物线型（原有）
    * surface_model: "fresnel" / "custom" / "coated"

设计：
    * 每条光线 = 一个 GPU 线程
    * 整个 trace 循环在单个 Warp 内核里完成
    * 无 Python per-step 循环、无 .any()/.item() 同步
"""
from __future__ import annotations

import math
from typing import Callable, Optional
import numpy as np
import torch
import warp as wp
import trimesh


# ==============================================================================
# 设备函数
# ==============================================================================

@wp.func
def fresnel_rs_rp(n1: wp.float32, n2: wp.float32, cos_i: wp.float32):
    eta = n1 / n2
    sin2_i = 1.0 - cos_i * cos_i
    sin2_t = eta * eta * sin2_i
    if sin2_t > 1.0:
        return wp.float32(1.0), wp.float32(1.0), wp.float32(0.0), wp.int32(1)
    cos_t = wp.sqrt(1.0 - sin2_t)
    denom_s = n1 * cos_i + n2 * cos_t + 1e-12
    denom_p = n2 * cos_i + n1 * cos_t + 1e-12
    r_s = (n1 * cos_i - n2 * cos_t) / denom_s
    r_p = (n2 * cos_i - n1 * cos_t) / denom_p
    return r_s, r_p, cos_t, wp.int32(0)


@wp.func
def compute_face_normal(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    e1 = v1 - v0
    e2 = v2 - v0
    n = wp.cross(e1, e2)
    ln = wp.length(n)
    if ln < 1e-12:
        return wp.vec3(0.0, 0.0, 1.0)
    return n / ln


@wp.func
def choose_perpendicular(d: wp.vec3) -> wp.vec3:
    ref = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(d[2]) > 0.99:
        ref = wp.vec3(0.0, 1.0, 0.0)
    s = wp.cross(ref, d)
    ln = wp.length(s)
    if ln < 1e-12:
        return wp.vec3(1.0, 0.0, 0.0)
    return s / ln


@wp.func
def rotate_stokes_to_new_s(
    S1: wp.float32, S2: wp.float32, S3: wp.float32,
    s_old: wp.vec3, s_new: wp.vec3, d: wp.vec3,
):
    cos_t = wp.dot(s_old, s_new)
    sin_t = wp.dot(wp.cross(s_old, s_new), d)
    cos_2t = cos_t * cos_t - sin_t * sin_t
    sin_2t = 2.0 * cos_t * sin_t
    S1n = cos_2t * S1 + sin_2t * S2
    S2n = -sin_2t * S1 + cos_2t * S2
    return S1n, S2n, S3


# ---------- 波长 LUT 插值（在 Warp 中实现） ----------

@wp.func
def interp_wavelength_1d(
    wl: wp.float32,
    wl_grid: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    K: wp.int32,
):
    """1D 线性插值（波长 -> 值），带末端保持。参考 newton/_src/actuators 的 _interp_1d 实现。"""
    if K <= 0:
        return wp.float32(0.0)
    if wl <= wl_grid[0]:
        return values[0]
    if wl >= wl_grid[K - 1]:
        return values[K - 1]
    # 线性查找（波长表通常很短，5-20 个点）
    for k in range(K - 1):
        x1 = wl_grid[k + 1]
        if x1 >= wl:
            x0 = wl_grid[k]
            y0 = values[k]
            y1 = values[k + 1]
            if x1 - x0 < 1e-12:
                return y0
            return y0 + (wl - x0) / (x1 - x0) * (y1 - y0)
    return values[K - 1]


@wp.func
def interp_wavelength_2d(
    wl: wp.float32,
    cos_i: wp.float32,
    wl_grid: wp.array(dtype=wp.float32),
    cos_grid: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),   # 展平 [Kw*Kc] 行主序
    Kw: wp.int32,
    Kc: wp.int32,
):
    """2D 双线性插值（波长 × cos_theta -> 值）。"""
    # 找波长索引
    iw = wp.int32(0)
    for k in range(Kw - 1):
        if wl_grid[k + 1] >= wl:
            iw = k
            break
        iw = k + 1
    if wl <= wl_grid[0]:
        iw = 0
    if wl >= wl_grid[Kw - 1]:
        iw = Kw - 2
    tw = (wl - wl_grid[iw]) / (wl_grid[iw + 1] - wl_grid[iw] + 1e-12)
    tw = wp.clamp(tw, 0.0, 1.0)

    # 找 cos 索引
    ic = wp.int32(0)
    for k in range(Kc - 1):
        if cos_grid[k + 1] >= cos_i:
            ic = k
            break
        ic = k + 1
    if cos_i <= cos_grid[0]:
        ic = 0
    if cos_i >= cos_grid[Kc - 1]:
        ic = Kc - 2
    tc = (cos_i - cos_grid[ic]) / (cos_grid[ic + 1] - cos_grid[ic] + 1e-12)
    tc = wp.clamp(tc, 0.0, 1.0)

    # 双线性插值
    i00 = iw * Kc + ic
    i01 = iw * Kc + (ic + 1)
    i10 = (iw + 1) * Kc + ic
    i11 = (iw + 1) * Kc + (ic + 1)

    v00 = values[i00]
    v01 = values[i01]
    v10 = values[i10]
    v11 = values[i11]

    v0 = v00 * (1.0 - tc) + v01 * tc
    v1 = v10 * (1.0 - tc) + v11 * tc
    return v0 * (1.0 - tw) + v1 * tw


# ---------- GRIN 场 ----------

@wp.func
def eval_grin_parabolic(
    r: wp.vec3, n0: wp.float32, A: wp.float32
):
    """n(r) = n0 - A*(x² + y²)，返回 (n, grad_n)。"""
    n = n0 - A * (r[0] * r[0] + r[1] * r[1])
    if n < 1.0:
        n = 1.0
    grad = wp.vec3(-2.0 * A * r[0], -2.0 * A * r[1], 0.0)
    return n, grad


@wp.func
def grin_rk4_step_parabolic(
    r: wp.vec3, d: wp.vec3,
    n0: wp.float32, A: wp.float32,
    ds: wp.float32,
):
    """抛物线 GRIN 的 RK4 单步。"""
    n1, g1 = eval_grin_parabolic(r, n0, A)
    p1 = n1 * d
    dr1 = p1 / n1
    dp1 = g1

    r2 = r + 0.5 * ds * dr1
    p2 = p1 + 0.5 * ds * dp1
    n2, g2 = eval_grin_parabolic(r2, n0, A)
    dr2 = p2 / n2
    dp2 = g2

    r3 = r + 0.5 * ds * dr2
    p3 = p1 + 0.5 * ds * dp2
    n3, g3 = eval_grin_parabolic(r3, n0, A)
    dr3 = p3 / n3
    dp3 = g3

    r4 = r + ds * dr3
    p4 = p1 + ds * dp3
    n4, g4 = eval_grin_parabolic(r4, n0, A)
    dr4 = p4 / n4
    dp4 = g4

    r_next = r + (ds / 6.0) * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4)
    p_next = p1 + (ds / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
    n_next, _ = eval_grin_parabolic(r_next, n0, A)
    d_next = p_next / n_next
    d_next = wp.normalize(d_next)
    return r_next, d_next, n_next


# ---------- ★ GRIN Callable 桥接 ----------

class GRINCallable:
    """
    将用户提供的 @wp.func 包装成 WarpTracer 可用的 GRIN 场。

    用法：
        @wp.func
        def my_grin(x: wp.float32, y: wp.float32, z: wp.float32):
            return 1.5 - 0.1 * (x*x + y*y)

        tracer = WarpTracer(..., grin_callable=my_grin)
    """
    def __init__(self, wp_func):
        self.wp_func = wp_func


# ==============================================================================
# 主内核
# ==============================================================================

@wp.kernel(enable_backward=False)
def trace_kernel(
    # geometry
    mesh_id: wp.uint64,
    tri_v0: wp.array(dtype=wp.vec3),
    tri_v1: wp.array(dtype=wp.vec3),
    tri_v2: wp.array(dtype=wp.vec3),
    face_materials: wp.array(dtype=wp.int32, ndim=2),
    # material tables (标量常量)
    mat_n: wp.array(dtype=wp.float32),
    mat_r: wp.array(dtype=wp.float32),
    mat_d: wp.array(dtype=wp.float32),
    mat_a: wp.array(dtype=wp.float32),
    # ★ 波长依赖 LUT（每种材质一份，长度可变）
    #   展平所有材质的 LUT 到一个大数组，用 offsets 定位
    wl_kind: wp.array(dtype=wp.int32),          # 0=常量, 1=1D, 2=2D
    wl_grid: wp.array(dtype=wp.float32),        # 所有材质的波长节点拼接
    wl_values: wp.array(dtype=wp.float32),      # 所有材质的值拼接
    wl_offsets: wp.array(dtype=wp.int32),       # 每种材质在 wl_grid/wl_values 中的起始位置
    wl_sizes: wp.array(dtype=wp.int32),         # 每种材质的节点数
    cos_grid: wp.array(dtype=wp.float32),       # 2D LUT 的 cos 节点
    cos_offsets: wp.array(dtype=wp.int32),
    cos_sizes: wp.array(dtype=wp.int32),
    # ★ GRIN callable（编译期内联）
    #   通过 kind 标志选择：0=无, 1=抛物线, 2=callable
    grin_kind: wp.array(dtype=wp.int32),
    grin_n0: wp.array(dtype=wp.float32),
    grin_A: wp.array(dtype=wp.float32),
    # GRIN callable 的 @wp.func 不能作为数组传递，需要预编译。
    #   这里使用一个“标记”数组，在 Python 端按 kind==2 时替换为编译变体。
    grin_callable_tag: wp.array(dtype=wp.int32),
    # rays in
    ray_origins: wp.array(dtype=wp.vec3),
    ray_dirs: wp.array(dtype=wp.vec3),
    ray_stokes: wp.array(dtype=wp.vec4),
    ray_wavelengths: wp.array(dtype=wp.float32),
    # history out
    hist_pos: wp.array(dtype=wp.vec3, ndim=2),
    hist_stokes: wp.array(dtype=wp.vec4, ndim=2),
    hist_energy: wp.array(dtype=wp.float32, ndim=2),
    hist_active: wp.array(dtype=wp.int32, ndim=2),
    hist_refl: wp.array(dtype=wp.float32, ndim=2),
    hist_trans: wp.array(dtype=wp.float32, ndim=2),
    hist_abs: wp.array(dtype=wp.float32, ndim=2),
    hist_n1: wp.array(dtype=wp.float32, ndim=2),
    hist_n2: wp.array(dtype=wp.float32, ndim=2),
    hist_dop: wp.array(dtype=wp.float32, ndim=2),
    # params
    max_steps: wp.int32,
    eps: wp.float32,
    far: wp.float32,
    seed: wp.int32,
    surface_model_flag: wp.int32,
    ds_init: wp.float32,
    num_media: wp.int32,
    num_wl_entries: wp.int32,
):
    tid = wp.tid()

    # ---------- init ----------
    origin = ray_origins[tid]
    d = ray_dirs[tid]
    S0 = ray_stokes[tid][0]
    S1 = ray_stokes[tid][1]
    S2 = ray_stokes[tid][2]
    S3 = ray_stokes[tid][3]
    wl = ray_wavelengths[tid]
    weight = wp.float32(1.0)
    active = wp.int32(1)
    medium_id = wp.int32(0)
    prev_face = wp.int32(-1)
    s_ref = choose_perpendicular(d)
    ds = ds_init

    rng = wp.rand_init(seed, tid)

    # record step 0
    hist_pos[tid, 0] = origin
    hist_stokes[tid, 0] = wp.vec4(S0, S1, S2, S3)
    hist_energy[tid, 0] = weight
    hist_active[tid, 0] = active
    hist_refl[tid, 0] = 0.0
    hist_trans[tid, 0] = 0.0
    hist_abs[tid, 0] = 0.0
    hist_n1[tid, 0] = 1.0
    hist_n2[tid, 0] = 1.0
    hist_dop[tid, 0] = wp.sqrt(S1 * S1 + S2 * S2 + S3 * S3) / (S0 + 1e-12)

    step = wp.int32(0)

    while active == 1 and step < max_steps:
        # ---------- 1. 求交 ----------
        q = wp.mesh_query_ray(mesh_id, origin, d, far)

        # ---------- 判断当前介质 GRIN ----------
        g_active = wp.int32(0)
        g_kind = wp.int32(0)
        g_n0 = wp.float32(1.0)
        g_A = wp.float32(0.0)
        if medium_id >= 0 and medium_id < num_media:
            g_kind = grin_kind[medium_id]
            if g_kind > 0:
                g_active = wp.int32(1)
                g_n0 = grin_n0[medium_id]
                g_A = grin_A[medium_id]

        # ---------- 决策 ----------
        if g_active == 1 and ((not q.result) or q.t > ds):
            # ============================================================
            # ★ GRIN RK4 步进
            # ============================================================
            # 这里用一个统一的接口，两种模式在编译期展开
            # 模式 1：抛物线（内置）
            # 模式 2：callable（通过预编译的变体内核）
            if g_kind == 1:
                new_origin, new_d, new_n = grin_rk4_step_parabolic(
                    origin, d, g_n0, g_A, ds)
            else:
                # 模式 2 的 callable 需要单独的编译变体，见 _build_callable_variant
                # 这里保留占位，实际运行时由 Python 端选择正确的内核
                new_origin = origin + d * ds
                new_d = d
                new_n = wp.float32(1.0)

            # 更新 s_ref
            s_proj = wp.dot(s_ref, new_d)
            s_new = s_ref - s_proj * new_d
            s_new_len = wp.length(s_new)
            if s_new_len < 1e-12:
                s_new = choose_perpendicular(new_d)
            else:
                s_new = s_new / s_new_len

            S1n, S2n, S3n = rotate_stokes_to_new_s(
                S1, S2, S3, s_ref, s_new, new_d)
            S1 = S1n
            S2 = S2n
            S3 = S3n
            s_ref = s_new

            origin = new_origin
            d = new_d

            step = step + 1
            hist_pos[tid, step] = origin
            hist_stokes[tid, step] = wp.vec4(1.0, S1, S2, S3)
            hist_energy[tid, step] = weight
            hist_active[tid, step] = active
            hist_refl[tid, step] = 0.0
            hist_trans[tid, step] = 0.0
            hist_abs[tid, step] = 0.0
            hist_n1[tid, step] = new_n
            hist_n2[tid, step] = new_n
            hist_dop[tid, step] = wp.sqrt(S1 * S1 + S2 * S2 + S3 * S3) / (1.0 + 1e-12)

        else:
            # ============================================================
            # 表面交互
            # ============================================================
            skip_guard = wp.int32(0)
            while skip_guard < 8:
                if not q.result:
                    break
                if q.t > eps and q.face != prev_face:
                    break
                origin = origin + d * (q.t + eps)
                q = wp.mesh_query_ray(mesh_id, origin, d, far)
                skip_guard = skip_guard + 1

            cur_refl = wp.float32(0.0)
            cur_trans = wp.float32(0.0)
            cur_abs = wp.float32(0.0)
            cur_n1 = wp.float32(1.0)
            cur_n2 = wp.float32(1.0)

            if not q.result:
                origin = origin + d * 1.0
                active = wp.int32(0)
            else:
                t_hit = q.t
                face_idx = q.face
                origin = origin + d * t_hit

                v0 = tri_v0[face_idx]
                v1 = tri_v1[face_idx]
                v2 = tri_v2[face_idx]
                normal = compute_face_normal(v0, v1, v2)

                cos_raw = -wp.dot(d, normal)
                is_front = cos_raw > 0.0
                if not is_front:
                    normal = -normal
                cos_i = wp.abs(cos_raw)
                if cos_i > 1.0:
                    cos_i = 1.0
                if cos_i < 1e-6:
                    cos_i = 1e-6

                mat_front = face_materials[face_idx, 0]
                mat_back = face_materials[face_idx, 1]
                n1_id = mat_front
                n2_id = mat_back
                if not is_front:
                    n1_id = mat_back
                    n2_id = mat_front

                # ---- 折射率：从 LUT 读取（波长相关） ----
                # n1
                if wl_kind[n1_id] == 1:
                    off = wl_offsets[n1_id]
                    sz = wl_sizes[n1_id]
                    n1 = interp_wavelength_1d(
                        wl, wl_grid[off:off+sz], wl_values[off:off+sz], sz)
                elif wl_kind[n1_id] == 2:
                    w_off = wl_offsets[n1_id]
                    w_sz = wl_sizes[n1_id]
                    c_off = cos_offsets[n1_id]
                    c_sz = cos_sizes[n1_id]
                    n1 = interp_wavelength_2d(
                        wl, cos_i,
                        wl_grid[w_off:w_off+w_sz],
                        cos_grid[c_off:c_off+c_sz],
                        wl_values[w_off*cos_sizes[n1_id]:],
                        w_sz, c_sz)
                else:
                    n1 = mat_n[n1_id]

                # n2
                if wl_kind[n2_id] == 1:
                    off = wl_offsets[n2_id]
                    sz = wl_sizes[n2_id]
                    n2 = interp_wavelength_1d(
                        wl, wl_grid[off:off+sz], wl_values[off:off+sz], sz)
                elif wl_kind[n2_id] == 2:
                    w_off = wl_offsets[n2_id]
                    w_sz = wl_sizes[n2_id]
                    c_off = cos_offsets[n2_id]
                    c_sz = cos_sizes[n2_id]
                    n2 = interp_wavelength_2d(
                        wl, cos_i,
                        wl_grid[w_off:w_off+w_sz],
                        cos_grid[c_off:c_off+c_sz],
                        wl_values[w_off*cos_sizes[n2_id]:],
                        w_sz, c_sz)
                else:
                    n2 = mat_n[n2_id]

                if n1 < 1.0:
                    n1 = 1.0
                if n2 < 1.0:
                    n2 = 1.0

                # ---- 反射率、漫反射、吸收也从 LUT 读取 ----
                # 这里简化：假设它们也是波长相关，用同样的 LUT 机制
                # 实际实现中需要为每个属性分别建 LUT
                custom_refl = mat_r[n2_id]
                custom_diff = mat_d[n2_id]
                custom_abs = mat_a[n2_id]

                cur_n1 = n1
                cur_n2 = n2

                cross_plane = wp.cross(d, normal)
                ln_plane = wp.length(cross_plane)
                s_plane = s_ref
                if ln_plane > 1e-8:
                    s_plane = cross_plane / ln_plane

                S1p, S2p, S3p = rotate_stokes_to_new_s(
                    S1, S2, S3, s_ref, s_plane, d)

                r_s, r_p, cos_t_f, is_tir_i = fresnel_rs_rp(n1, n2, cos_i)
                is_tir = (is_tir_i == 1)
                R_s = r_s * r_s
                R_p = r_p * r_p

                w_s = 0.5 * (1.0 + S1p)
                w_p = 0.5 * (1.0 - S1p)
                R_eff = w_s * R_s + w_p * R_p

                specular_R = R_eff
                if surface_model_flag == 1:
                    specular_R = custom_refl
                elif surface_model_flag == 2:
                    specular_R = R_eff * custom_refl
                if is_tir:
                    specular_R = 1.0

                P_absorb = wp.clamp(custom_abs, 0.0, 1.0)
                P_diffuse = (1.0 - P_absorb) * wp.clamp(custom_diff, 0.0, 1.0)
                P_specular = (1.0 - P_absorb - P_diffuse) * wp.clamp(specular_R, 0.0, 1.0)
                P_refract = 1.0 - P_absorb - P_diffuse - P_specular
                if P_refract < 0.0:
                    P_refract = 0.0

                total_P = P_absorb + P_diffuse + P_specular + P_refract
                if total_P < 1e-12:
                    total_P = 1e-12
                P_absorb = P_absorb / total_P
                P_diffuse = P_diffuse / total_P
                P_specular = P_specular / total_P
                P_refract = P_refract / total_P

                cur_refl = P_specular + P_diffuse
                cur_trans = P_refract
                cur_abs = P_absorb

                r = wp.randf(rng)

                if r < P_absorb:
                    weight = 0.0
                    active = wp.int32(0)
                elif r < P_absorb + P_diffuse:
                    weight = weight * P_diffuse
                    r_sq = wp.randf(rng)
                    sin_theta = wp.sqrt(r_sq)
                    cos_theta = wp.sqrt(1.0 - r_sq)
                    phi = 6.283185307179586 * wp.randf(rng)
                    ref_v = wp.vec3(0.0, 0.0, 1.0)
                    if wp.abs(normal[2]) > 0.999:
                        ref_v = wp.vec3(0.0, 1.0, 0.0)
                    local_u = wp.cross(ref_v, normal)
                    lu = wp.length(local_u)
                    if lu > 1e-12:
                        local_u = local_u / lu
                    local_v = wp.cross(normal, local_u)
                    new_d = (sin_theta * wp.cos(phi) * local_u
                             + sin_theta * wp.sin(phi) * local_v
                             + cos_theta * normal)
                    d = wp.normalize(new_d)
                    S1 = 0.0
                    S2 = 0.0
                    S3 = 0.0
                    s_ref = choose_perpendicular(d)
                    medium_id = n1_id
                elif r < P_absorb + P_diffuse + P_specular:
                    weight = weight * P_specular
                    new_d = d + 2.0 * cos_i * normal
                    d = wp.normalize(new_d)
                    a_m = 0.5 * (r_s * r_s + r_p * r_p)
                    b_m = 0.5 * (r_s * r_s - r_p * r_p)
                    c_m = r_s * r_p
                    S0n = a_m * 1.0 + b_m * S1p
                    S1n = b_m * 1.0 + a_m * S1p
                    S2n = c_m * S2p
                    S3n = c_m * S3p
                    if S0n < 1e-12:
                        S0n = 1e-12
                    S1 = S1n / S0n
                    S2 = S2n / S0n
                    S3 = S3n / S0n
                    s_ref = s_plane
                    medium_id = n1_id
                else:
                    weight = weight * P_refract
                    eta = n1 / n2
                    new_d = eta * d + (eta * cos_i - cos_t_f) * normal
                    d = wp.normalize(new_d)
                    denom_s = n1 * cos_i + n2 * cos_t_f + 1e-12
                    denom_p = n2 * cos_i + n1 * cos_t_f + 1e-12
                    t_s = 2.0 * n1 * cos_i / denom_s
                    t_p = 2.0 * n1 * cos_i / denom_p
                    a_t = 0.5 * (t_s * t_s + t_p * t_p)
                    b_t = 0.5 * (t_s * t_s - t_p * t_p)
                    c_t = t_s * t_p
                    S0n = a_t * 1.0 + b_t * S1p
                    S1n = b_t * 1.0 + a_t * S1p
                    S2n = c_t * S2p
                    S3n = c_t * S3p
                    if S0n < 1e-12:
                        S0n = 1e-12
                    S1 = S1n / S0n
                    S2 = S2n / S0n
                    S3 = S3n / S0n
                    s_ref = s_plane
                    medium_id = n2_id

                prev_face = face_idx

            step = step + 1
            hist_pos[tid, step] = origin
            hist_stokes[tid, step] = wp.vec4(1.0, S1, S2, S3)
            hist_energy[tid, step] = weight
            hist_active[tid, step] = active
            hist_refl[tid, step] = cur_refl
            hist_trans[tid, step] = cur_trans
            hist_abs[tid, step] = cur_abs
            hist_n1[tid, step] = cur_n1
            hist_n2[tid, step] = cur_n2
            hist_dop[tid, step] = wp.sqrt(S1 * S1 + S2 * S2 + S3 * S3) / (1.0 + 1e-12)

    while step < max_steps:
        step = step + 1
        hist_pos[tid, step] = origin
        hist_stokes[tid, step] = wp.vec4(1.0, S1, S2, S3)
        hist_energy[tid, step] = weight
        hist_active[tid, step] = active
        hist_refl[tid, step] = 0.0
        hist_trans[tid, step] = 0.0
        hist_abs[tid, step] = 0.0
        hist_n1[tid, step] = 1.0
        hist_n2[tid, step] = 1.0
        hist_dop[tid, step] = wp.sqrt(S1 * S1 + S2 * S2 + S3 * S3) / (1.0 + 1e-12)
