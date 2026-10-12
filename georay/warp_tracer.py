# georay/warp_tracer.py
"""
全 Warp 内核实现的 GeoRay tracer。

设计：
    * 每条光线 = 一个 GPU 线程
    * 整个 trace 循环（max_steps 步）在单个 Warp 内核里完成
    * 无 Python per-step 循环、无 .any()/.item() 同步
    * history 预分配为固定 shape [N, S+1, ...]

物理：
    * Fresnel (s/p 偏振加权)
    * TIR
    * Stokes 偏振追踪 (Jones-Mueller + 基准旋转)
    * Monte Carlo 分支 (reflect / refract / diffuse / absorb)
    * 能量权重累积
    * Diffuse 半球采样
    * surface_model 支持 "fresnel" / "custom" / "coated"

限制：
    * 不支持 GRIN 介质（后续可扩展）
    * 不支持波长相关材质
"""
from __future__ import annotations

import math
import numpy as np
import torch
import warp as wp
import trimesh


# ==============================================================================
# 设备函数
# ==============================================================================

@wp.func
def fresnel_rs_rp(n1: wp.float32, n2: wp.float32, cos_i: wp.float32):
    """返回 (r_s, r_p, cos_t, is_tir)."""
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
    """把 (S1,S2,S3) 从 s_old 基准旋转到 s_new 基准。"""
    cos_t = wp.dot(s_old, s_new)
    sin_t = wp.dot(wp.cross(s_old, s_new), d)
    cos_2t = cos_t * cos_t - sin_t * sin_t
    sin_2t = 2.0 * cos_t * sin_t
    S1n = cos_2t * S1 + sin_2t * S2
    S2n = -sin_2t * S1 + cos_2t * S2
    return S1n, S2n, S3


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
    face_materials: wp.array(dtype=wp.int32, ndim=2),   # [M, 2]
    # material tables
    mat_n: wp.array(dtype=wp.float32),
    mat_r: wp.array(dtype=wp.float32),
    mat_d: wp.array(dtype=wp.float32),
    mat_a: wp.array(dtype=wp.float32),
    # rays in
    ray_origins: wp.array(dtype=wp.vec3),
    ray_dirs: wp.array(dtype=wp.vec3),
    ray_stokes: wp.array(dtype=wp.vec4),
    # history out
    hist_pos: wp.array(dtype=wp.vec3, ndim=2),          # [N, S+1]
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
    surface_model_flag: wp.int32,   # ★ 0=fresnel, 1=custom, 2=coated
):
    tid = wp.tid()

    # ---------- init ----------
    origin = ray_origins[tid]
    d = ray_dirs[tid]
    S0 = ray_stokes[tid][0]
    S1 = ray_stokes[tid][1]
    S2 = ray_stokes[tid][2]
    S3 = ray_stokes[tid][3]
    weight = wp.float32(1.0)
    active = wp.int32(1)
    medium_id = wp.int32(0)
    prev_face = wp.int32(-1)
    s_ref = choose_perpendicular(d)

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
        # ---------- 1. ray cast ----------
        q = wp.mesh_query_ray(mesh_id, origin, d, far)

        cur_refl = wp.float32(0.0)
        cur_trans = wp.float32(0.0)
        cur_abs = wp.float32(0.0)
        cur_n1 = wp.float32(1.0)
        cur_n2 = wp.float32(1.0)

        if not q.result:
            # escape
            origin = origin + d * 1.0
            active = wp.int32(0)
        else:
            t_hit = q.t
            face_idx = q.face

            if t_hit <= eps or face_idx == prev_face:
                # skip self-intersection / same face
                origin = origin + d * (t_hit + eps)
            else:
                hit_pt = origin + d * t_hit
                origin = hit_pt

                # ---------- 2. face normal ----------
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

                # ---------- 3. materials ----------
                mat_front = face_materials[face_idx, 0]
                mat_back = face_materials[face_idx, 1]
                n1_id = mat_front
                n2_id = mat_back
                if not is_front:
                    n1_id = mat_back
                    n2_id = mat_front

                n1 = mat_n[n1_id]
                n2 = mat_n[n2_id]
                if n1 < 1.0:
                    n1 = 1.0
                if n2 < 1.0:
                    n2 = 1.0

                custom_refl = mat_r[n2_id]
                custom_diff = mat_d[n2_id]
                custom_abs = mat_a[n2_id]

                cur_n1 = n1
                cur_n2 = n2

                # ---------- 4. plane-of-incidence s basis ----------
                cross_plane = wp.cross(d, normal)
                ln_plane = wp.length(cross_plane)
                s_plane = s_ref
                if ln_plane > 1e-8:
                    s_plane = cross_plane / ln_plane

                # rotate stokes from s_ref to s_plane
                S1p, S2p, S3p = rotate_stokes_to_new_s(
                    S1, S2, S3, s_ref, s_plane, d)

                # ---------- 5. Fresnel ----------
                r_s, r_p, cos_t_f, is_tir_i = fresnel_rs_rp(n1, n2, cos_i)
                is_tir = (is_tir_i == 1)
                R_s = r_s * r_s
                R_p = r_p * r_p

                w_s = 0.5 * (1.0 + S1p)
                w_p = 0.5 * (1.0 - S1p)
                R_eff = w_s * R_s + w_p * R_p

                # ★ surface_model: 0=fresnel, 1=custom, 2=coated
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

                # ---------- 6. Monte Carlo branching ----------
                r = wp.randf(rng)

                if r < P_absorb:
                    # absorbed
                    weight = 0.0
                    active = wp.int32(0)

                elif r < P_absorb + P_diffuse:
                    # diffuse
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
                    new_d = wp.normalize(new_d)
                    d = new_d
                    S1 = 0.0
                    S2 = 0.0
                    S3 = 0.0
                    s_ref = choose_perpendicular(d)
                    medium_id = n1_id

                elif r < P_absorb + P_diffuse + P_specular:
                    # specular reflect
                    weight = weight * P_specular

                    new_d = d + 2.0 * cos_i * normal
                    new_d = wp.normalize(new_d)
                    d = new_d

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
                    # refract
                    weight = weight * P_refract

                    eta = n1 / n2
                    new_d = eta * d + (eta * cos_i - cos_t_f) * normal
                    new_d = wp.normalize(new_d)
                    d = new_d

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

        # ---------- 7. record step ----------
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

    # ---------- 8. fill remaining steps ----------
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


# ==============================================================================
# Python 包装类
# ==============================================================================

class WarpTracer:
    """全 Warp 内核实现的 tracer，接口与 AdvancedRayTracerPBRTGPU 兼容。"""

    def __init__(self, vertices, faces, face_materials, materials_cfg, device):
        self.device = device
        if str(device).startswith("cuda"):
            self.warp_device = "cuda:0"
        else:
            self.warp_device = "cpu"

        # ---- 确保 Warp 已初始化（wp.init() 幂等，不需要 is_initialized 检查） ----
        wp.init()

        # ---- 转 numpy ----
        verts_np = vertices.detach().cpu().numpy().astype(np.float32)
        faces_np = faces.detach().cpu().numpy().astype(np.int32).flatten()

        # ---- 构建 Warp mesh ----
        self.mesh = wp.Mesh(
            points=wp.array(verts_np, dtype=wp.vec3, device=self.warp_device),
            indices=wp.array(faces_np, dtype=wp.int32, device=self.warp_device),
        )
        self.mesh_id = self.mesh.id

        # ---- 三角形顶点数组 (用于计算法线) ----
        faces_long = faces.detach().cpu().numpy().astype(np.int64)
        tris = verts_np[faces_long]                # [M, 3, 3]
        tri_v0 = np.ascontiguousarray(tris[:, 0]).astype(np.float32)
        tri_v1 = np.ascontiguousarray(tris[:, 1]).astype(np.float32)
        tri_v2 = np.ascontiguousarray(tris[:, 2]).astype(np.float32)
        self.tri_v0 = wp.array(tri_v0, dtype=wp.vec3, device=self.warp_device)
        self.tri_v1 = wp.array(tri_v1, dtype=wp.vec3, device=self.warp_device)
        self.tri_v2 = wp.array(tri_v2, dtype=wp.vec3, device=self.warp_device)

        # ---- face_materials ----
        fm_np = face_materials.detach().cpu().numpy().astype(np.int32)
        self.face_materials = wp.array(
            fm_np, dtype=wp.int32, ndim=2, device=self.warp_device)

        # ---- 材质标量查找表 ----
        max_mat = max(materials_cfg.keys()) + 1
        n_arr = np.ones(max_mat, dtype=np.float32)
        r_arr = np.zeros(max_mat, dtype=np.float32)
        d_arr = np.zeros(max_mat, dtype=np.float32)
        a_arr = np.zeros(max_mat, dtype=np.float32)
        for k, cfg in materials_cfg.items():
            v_n = cfg.get("n", 1.0)
            if not isinstance(v_n, (int, float)):
                raise NotImplementedError(
                    f"WarpTracer only supports scalar material properties. "
                    f"Material {k} has n={v_n}")
            n_arr[k] = float(v_n)
            r_arr[k] = float(cfg.get("reflectivity", 0.0))
            d_arr[k] = float(cfg.get("diffuse", 0.0))
            a_arr[k] = float(cfg.get("absorptivity", 0.0))

        self.mat_n = wp.array(n_arr, dtype=wp.float32, device=self.warp_device)
        self.mat_r = wp.array(r_arr, dtype=wp.float32, device=self.warp_device)
        self.mat_d = wp.array(d_arr, dtype=wp.float32, device=self.warp_device)
        self.mat_a = wp.array(a_arr, dtype=wp.float32, device=self.warp_device)

    def trace(self, ray_dict, max_steps=50, ds_init=0.02, ds_min=1e-3,
              ds_max=0.1, rk_tol=1e-4, fresnel=True, verbose=False,
              surface_model=None, rr_threshold=0.0, seed=42):
        """
        追踪光线。

        Args:
            ray_dict: generate_rays() 的输出
            max_steps: 最大步数
            fresnel: legacy flag；surface_model=None 时生效
            surface_model: "fresnel" | "custom" | "coated"
            seed: 随机种子
            其他参数忽略（保留接口兼容性）

        Returns:
            dict，键与 AdvancedRayTracerPBRTGPU.trace 一致。
        """
        # ---- 解析 surface_model（与 PyTorch 版本语义一致） ----
        if surface_model is None:
            surface_model = "coated" if fresnel else "custom"
        if surface_model not in ("fresnel", "custom", "coated"):
            raise ValueError(
                f"surface_model must be 'fresnel', 'custom', or 'coated', "
                f"got '{surface_model}'")
        model_map = {"fresnel": 0, "custom": 1, "coated": 2}
        surface_model_flag = model_map[surface_model]

        origins = ray_dict["origins"].to(self.device).contiguous().float()
        dirs = ray_dict["dirs"].to(self.device).contiguous().float()
        stokes = ray_dict["stokes"].to(self.device).contiguous().float()

        N = origins.shape[0]
        S = max_steps + 1

        # ---- 转换到 Warp 数组 ----
        wp_origins = wp.from_torch(origins, dtype=wp.vec3)
        wp_dirs = wp.from_torch(dirs, dtype=wp.vec3)
        wp_stokes = wp.from_torch(stokes, dtype=wp.vec4)

        # ---- 预分配输出 ----
        wp_pos = wp.zeros((N, S), dtype=wp.vec3, device=self.warp_device)
        wp_stokes_hist = wp.zeros((N, S), dtype=wp.vec4, device=self.warp_device)
        wp_energy = wp.zeros((N, S), dtype=wp.float32, device=self.warp_device)
        wp_active = wp.zeros((N, S), dtype=wp.int32, device=self.warp_device)
        wp_refl = wp.zeros((N, S), dtype=wp.float32, device=self.warp_device)
        wp_trans = wp.zeros((N, S), dtype=wp.float32, device=self.warp_device)
        wp_abs = wp.zeros((N, S), dtype=wp.float32, device=self.warp_device)
        wp_n1 = wp.zeros((N, S), dtype=wp.float32, device=self.warp_device)
        wp_n2 = wp.zeros((N, S), dtype=wp.float32, device=self.warp_device)
        wp_dop = wp.zeros((N, S), dtype=wp.float32, device=self.warp_device)

        # ---- 启动内核 ----
        wp.launch(
            trace_kernel,
            dim=N,
            inputs=[
                self.mesh_id,
                self.tri_v0, self.tri_v1, self.tri_v2,
                self.face_materials,
                self.mat_n, self.mat_r, self.mat_d, self.mat_a,
                wp_origins, wp_dirs, wp_stokes,
                wp_pos, wp_stokes_hist, wp_energy, wp_active,
                wp_refl, wp_trans, wp_abs, wp_n1, wp_n2, wp_dop,
                max_steps,
                np.float32(1e-4),
                np.float32(1e10),
                int(seed),
                int(surface_model_flag),     # ★ 新增
            ],
            device=self.warp_device,
        )
        wp.synchronize()

        # ---- 转换回 torch ----
        pos_t = wp.to_torch(wp_pos).clone()
        stokes_t = wp.to_torch(wp_stokes_hist).clone().view(N, S, 4)
        energy_t = wp.to_torch(wp_energy).clone()
        active_t = wp.to_torch(wp_active).clone().bool()
        refl_t = wp.to_torch(wp_refl).clone().unsqueeze(-1)
        trans_t = wp.to_torch(wp_trans).clone().unsqueeze(-1)
        abs_t = wp.to_torch(wp_abs).clone().unsqueeze(-1)
        n1_t = wp.to_torch(wp_n1).clone().unsqueeze(-1)
        n2_t = wp.to_torch(wp_n2).clone().unsqueeze(-1)
        dop_t = wp.to_torch(wp_dop).clone().unsqueeze(-1)

        # energy_conserved: 简化判断
        energy_conserved = bool(
            torch.all((energy_t >= -1e-6) & (energy_t <= 1.0 + 1e-6)).item())

        return {
            "positions": pos_t,
            "stokes": stokes_t,
            "active": active_t,
            "wavelengths": ray_dict["wavelengths"],
            "reflectivity": refl_t,
            "transmissivity": trans_t,
            "absorptivity": abs_t,
            "n1": n1_t,
            "n2": n2_t,
            "polarization_deg": torch.zeros_like(refl_t),
            "dop": dop_t,
            "energy": energy_t,
            "rk_error": torch.zeros((N, S), device=self.device),
            "ds_history": torch.zeros((N, S), device=self.device),
            "energy_conserved": energy_conserved,
        }
