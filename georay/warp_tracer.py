# georay/warp_tracer.py
"""
全 Warp 内核实现的 GeoRay tracer（callable GRIN 支持版）。

支持：
    * Fresnel (s/p 偏振加权) + TIR
    * Stokes 偏振追踪（Jones-Mueller + 基准旋转）
    * Monte Carlo 分支（reflect / refract / diffuse / absorb）
    * 能量权重累积
    * Diffuse 半球采样
    * surface_model: "fresnel" / "custom" / "coated"

属性 LUT（n / reflectivity / diffuse / absorptivity）：
    * kind=0: 标量常量
    * kind=1: 1D 波长 LUT
    * kind=2: 2D 波长×角度 LUT
    * kind=3: 3D GRIN LUT
    * kind=4: 4D GRIN+波长 LUT

GRIN 场：
    * parabolic 内置 RK4
    * callable Python 函数 → 自动采样为 3D grid → 走 grid 路径
    * 3D dict-grid  三线性插值 + 解析梯度
    * 4D dict-grid  四线性插值 + 中心差分梯度
"""

from __future__ import annotations

import math
import numpy as np
import torch
import warp as wp
import trimesh
from .exceptions import BackendError, InvalidRayError
from .tracer import _validate_materials_cfg, _validate_ray_dict


# ==============================================================================
# 基础设备函数
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
        ref = wp.vec3(1.0, 0.0, 0.0)
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


# ==============================================================================
# LUT 插值函数
# ==============================================================================

@wp.func
def interp_1d_full(
    x: wp.float32,
    grid: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    offset: wp.int32,
    K: wp.int32,
):
    if K <= 0:
        return wp.float32(0.0)
    if K == 1:
        return values[offset]

    x0_g = grid[offset]
    xK_g = grid[offset + K - 1]

    if x <= x0_g:
        return values[offset]
    if x >= xK_g:
        return values[offset + K - 1]

    result = values[offset + K - 1]
    found = wp.int32(0)

    for k in range(256):
        if found == 0 and k < K - 1:
            x1 = grid[offset + k + 1]
            if x1 >= x:
                x0 = grid[offset + k]
                y0 = values[offset + k]
                y1 = values[offset + k + 1]
                denom = x1 - x0
                if denom < 1e-12:
                    result = y0
                else:
                    t = (x - x0) / denom
                    result = y0 + t * (y1 - y0)
                found = wp.int32(1)
    return result


@wp.func
def find_interval_1d(
    x: wp.float32,
    grid: wp.array(dtype=wp.float32),
    offset: wp.int32,
    K: wp.int32,
):
    if K <= 1:
        return wp.int32(0), wp.float32(0.0)

    x0_g = grid[offset]
    xK_g = grid[offset + K - 1]

    if x <= x0_g:
        return wp.int32(0), wp.float32(0.0)
    if x >= xK_g:
        return wp.int32(K - 2), wp.float32(1.0)

    idx_out = wp.int32(K - 2)
    t_out = wp.float32(1.0)
    found = wp.int32(0)

    for k in range(256):
        if found == 0 and k < K - 1:
            x1 = grid[offset + k + 1]
            if x1 >= x:
                x0 = grid[offset + k]
                denom = x1 - x0
                if denom < 1e-12:
                    idx_out = k
                    t_out = wp.float32(0.0)
                else:
                    idx_out = k
                    t_out = (x - x0) / denom
                found = wp.int32(1)

    return idx_out, t_out


@wp.func
def interp_2d_full(
    x: wp.float32, y: wp.float32,
    x_grid: wp.array(dtype=wp.float32),
    y_grid: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    x_off: wp.int32, x_size: wp.int32,
    y_off: wp.int32, y_size: wp.int32,
    val_off: wp.int32,
):
    if x_size <= 1 or y_size <= 1:
        if x_size > 0 and y_size > 0:
            return values[val_off]
        return wp.float32(0.0)

    ix, tx = find_interval_1d(x, x_grid, x_off, x_size)
    iy, ty = find_interval_1d(y, y_grid, y_off, y_size)

    i00 = val_off + ix * y_size + iy
    i01 = i00 + 1
    i10 = val_off + (ix + 1) * y_size + iy
    i11 = i10 + 1

    v00 = values[i00]
    v01 = values[i01]
    v10 = values[i10]
    v11 = values[i11]

    v0 = v00 * (1.0 - ty) + v01 * ty
    v1 = v10 * (1.0 - ty) + v11 * ty
    return v0 * (1.0 - tx) + v1 * tx


@wp.func
def interp_3d_grin(
    x: wp.float32, y: wp.float32, z: wp.float32,
    gx: wp.array(dtype=wp.float32),
    gy: wp.array(dtype=wp.float32),
    gz: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    x_off: wp.int32, x_size: wp.int32,
    y_off: wp.int32, y_size: wp.int32,
    z_off: wp.int32, z_size: wp.int32,
    val_off: wp.int32,
):
    if x_size <= 1 or y_size <= 1 or z_size <= 1:
        if x_size > 0 and y_size > 0 and z_size > 0:
            return values[val_off]
        return wp.float32(0.0)

    ix, tx = find_interval_1d(x, gx, x_off, x_size)
    iy, ty = find_interval_1d(y, gy, y_off, y_size)
    iz, tz = find_interval_1d(z, gz, z_off, z_size)

    yz = y_size * z_size
    base_ix = val_off + ix * yz
    base_ix1 = val_off + (ix + 1) * yz

    i_y0_z0 = base_ix + iy * z_size + iz
    i_y0_z1 = i_y0_z0 + 1
    i_y1_z0 = base_ix + (iy + 1) * z_size + iz
    i_y1_z1 = i_y1_z0 + 1

    i_ix1_y0_z0 = base_ix1 + iy * z_size + iz
    i_ix1_y0_z1 = i_ix1_y0_z0 + 1
    i_ix1_y1_z0 = base_ix1 + (iy + 1) * z_size + iz
    i_ix1_y1_z1 = i_ix1_y1_z0 + 1

    v000 = values[i_y0_z0]
    v001 = values[i_y0_z1]
    v010 = values[i_y1_z0]
    v011 = values[i_y1_z1]
    v100 = values[i_ix1_y0_z0]
    v101 = values[i_ix1_y0_z1]
    v110 = values[i_ix1_y1_z0]
    v111 = values[i_ix1_y1_z1]

    v00 = v000 * (1.0 - tz) + v001 * tz
    v01 = v010 * (1.0 - tz) + v011 * tz
    v10 = v100 * (1.0 - tz) + v101 * tz
    v11 = v110 * (1.0 - tz) + v111 * tz

    v0 = v00 * (1.0 - ty) + v01 * ty
    v1 = v10 * (1.0 - ty) + v11 * ty

    return v0 * (1.0 - tx) + v1 * tx


@wp.func
def interp_3d_grin_and_grad(
    x: wp.float32, y: wp.float32, z: wp.float32,
    gx: wp.array(dtype=wp.float32),
    gy: wp.array(dtype=wp.float32),
    gz: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    x_off: wp.int32, x_size: wp.int32,
    y_off: wp.int32, y_size: wp.int32,
    z_off: wp.int32, z_size: wp.int32,
    val_off: wp.int32,
):
    n_val = interp_3d_grin(x, y, z, gx, gy, gz, values,
                            x_off, x_size, y_off, y_size, z_off, z_size, val_off)

    if x_size <= 1 or y_size <= 1 or z_size <= 1:
        return n_val, wp.vec3(0.0, 0.0, 0.0)

    ix, tx = find_interval_1d(x, gx, x_off, x_size)
    iy, ty = find_interval_1d(y, gy, y_off, y_size)
    iz, tz = find_interval_1d(z, gz, z_off, z_size)

    yz = y_size * z_size
    base_ix = val_off + ix * yz
    base_ix1 = val_off + (ix + 1) * yz

    i000 = base_ix + iy * z_size + iz
    i001 = i000 + 1
    i010 = base_ix + (iy + 1) * z_size + iz
    i011 = i010 + 1

    i100 = base_ix1 + iy * z_size + iz
    i101 = i100 + 1
    i110 = base_ix1 + (iy + 1) * z_size + iz
    i111 = i110 + 1

    v000 = values[i000]
    v001 = values[i001]
    v010 = values[i010]
    v011 = values[i011]
    v100 = values[i100]
    v101 = values[i101]
    v110 = values[i110]
    v111 = values[i111]

    v0_lo_x = (v000 * (1.0 - tz) + v001 * tz) * (1.0 - ty) + \
              (v010 * (1.0 - tz) + v011 * tz) * ty
    v0_hi_x = (v100 * (1.0 - tz) + v101 * tz) * (1.0 - ty) + \
              (v110 * (1.0 - tz) + v111 * tz) * ty
    dx = gx[x_off + ix + 1] - gx[x_off + ix]
    dndx = (v0_hi_x - v0_lo_x) / (dx + 1e-12)

    v0_lo_y = (v000 * (1.0 - tx) + v100 * tx) * (1.0 - tz) + \
              (v001 * (1.0 - tx) + v101 * tx) * tz
    v0_hi_y = (v010 * (1.0 - tx) + v110 * tx) * (1.0 - tz) + \
              (v011 * (1.0 - tx) + v111 * tx) * tz
    dy = gy[y_off + iy + 1] - gy[y_off + iy]
    dndy = (v0_hi_y - v0_lo_y) / (dy + 1e-12)

    v0_lo_z = (v000 * (1.0 - tx) + v100 * tx) * (1.0 - ty) + \
              (v010 * (1.0 - tx) + v110 * tx) * ty
    v0_hi_z = (v001 * (1.0 - tx) + v101 * tx) * (1.0 - ty) + \
              (v011 * (1.0 - tx) + v111 * tx) * ty
    dz = gz[z_off + iz + 1] - gz[z_off + iz]
    dndz = (v0_hi_z - v0_lo_z) / (dz + 1e-12)

    return n_val, wp.vec3(dndx, dndy, dndz)


@wp.func
def interp_4d_grin_wl(
    x: wp.float32, y: wp.float32, z: wp.float32, wl: wp.float32,
    gx: wp.array(dtype=wp.float32),
    gy: wp.array(dtype=wp.float32),
    gz: wp.array(dtype=wp.float32),
    gwl: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    x_off: wp.int32, x_size: wp.int32,
    y_off: wp.int32, y_size: wp.int32,
    z_off: wp.int32, z_size: wp.int32,
    wl_off: wp.int32, wl_size: wp.int32,
    val_off: wp.int32,
):
    if (x_size <= 1 or y_size <= 1 or z_size <= 1 or wl_size <= 1):
        if x_size > 0 and y_size > 0 and z_size > 0 and wl_size > 0:
            return values[val_off]
        return wp.float32(0.0)

    ix, tx = find_interval_1d(x, gx, x_off, x_size)
    iy, ty = find_interval_1d(y, gy, y_off, y_size)
    iz, tz = find_interval_1d(z, gz, z_off, z_size)
    iw, tw = find_interval_1d(wl, gwl, wl_off, wl_size)

    yzw = y_size * z_size * wl_size
    zw = z_size * wl_size

    base_ix = val_off + ix * yzw
    base_ix1 = val_off + (ix + 1) * yzw

    base_00 = base_ix + iy * zw + iz * wl_size + iw
    v_z0_w0 = values[base_00]
    v_z0_w1 = values[base_00 + 1]
    v_z1_w0 = values[base_00 + wl_size]
    v_z1_w1 = values[base_00 + wl_size + 1]
    v_z0 = v_z0_w0 * (1.0 - tw) + v_z0_w1 * tw
    v_z1 = v_z1_w0 * (1.0 - tw) + v_z1_w1 * tw
    v00 = v_z0 * (1.0 - tz) + v_z1 * tz

    base_01 = base_ix + (iy + 1) * zw + iz * wl_size + iw
    v_z0_w0 = values[base_01]
    v_z0_w1 = values[base_01 + 1]
    v_z1_w0 = values[base_01 + wl_size]
    v_z1_w1 = values[base_01 + wl_size + 1]
    v_z0 = v_z0_w0 * (1.0 - tw) + v_z0_w1 * tw
    v_z1 = v_z1_w0 * (1.0 - tw) + v_z1_w1 * tw
    v01 = v_z0 * (1.0 - tz) + v_z1 * tz

    base_10 = base_ix1 + iy * zw + iz * wl_size + iw
    v_z0_w0 = values[base_10]
    v_z0_w1 = values[base_10 + 1]
    v_z1_w0 = values[base_10 + wl_size]
    v_z1_w1 = values[base_10 + wl_size + 1]
    v_z0 = v_z0_w0 * (1.0 - tw) + v_z0_w1 * tw
    v_z1 = v_z1_w0 * (1.0 - tw) + v_z1_w1 * tw
    v10 = v_z0 * (1.0 - tz) + v_z1 * tz

    base_11 = base_ix1 + (iy + 1) * zw + iz * wl_size + iw
    v_z0_w0 = values[base_11]
    v_z0_w1 = values[base_11 + 1]
    v_z1_w0 = values[base_11 + wl_size]
    v_z1_w1 = values[base_11 + wl_size + 1]
    v_z0 = v_z0_w0 * (1.0 - tw) + v_z0_w1 * tw
    v_z1 = v_z1_w0 * (1.0 - tw) + v_z1_w1 * tw
    v11 = v_z0 * (1.0 - tz) + v_z1 * tz

    v0 = v00 * (1.0 - ty) + v01 * ty
    v1 = v10 * (1.0 - ty) + v11 * ty
    return v0 * (1.0 - tx) + v1 * tx


@wp.func
def interp_4d_grin_wl_and_grad(
    x: wp.float32, y: wp.float32, z: wp.float32, wl: wp.float32,
    gx: wp.array(dtype=wp.float32),
    gy: wp.array(dtype=wp.float32),
    gz: wp.array(dtype=wp.float32),
    gwl: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    x_off: wp.int32, x_size: wp.int32,
    y_off: wp.int32, y_size: wp.int32,
    z_off: wp.int32, z_size: wp.int32,
    wl_off: wp.int32, wl_size: wp.int32,
    val_off: wp.int32,
):
    n_val = interp_4d_grin_wl(x, y, z, wl, gx, gy, gz, gwl, values,
                                x_off, x_size, y_off, y_size,
                                z_off, z_size, wl_off, wl_size, val_off)

    h = wp.float32(1e-3)
    if x_size > 1:
        dx = (gx[x_off + x_size - 1] - gx[x_off]) / wp.float32(x_size - 1)
        h = wp.min(h, dx * 0.1)
    if y_size > 1:
        dy = (gy[y_off + y_size - 1] - gy[y_off]) / wp.float32(y_size - 1)
        h = wp.min(h, dy * 0.1)
    if z_size > 1:
        dz = (gz[z_off + z_size - 1] - gz[z_off]) / wp.float32(z_size - 1)
        h = wp.min(h, dz * 0.1)
    h = wp.max(h, wp.float32(1e-5))

    nx_p = interp_4d_grin_wl(x + h, y, z, wl, gx, gy, gz, gwl, values,
                              x_off, x_size, y_off, y_size, z_off, z_size,
                              wl_off, wl_size, val_off)
    nx_m = interp_4d_grin_wl(x - h, y, z, wl, gx, gy, gz, gwl, values,
                              x_off, x_size, y_off, y_size, z_off, z_size,
                              wl_off, wl_size, val_off)
    ny_p = interp_4d_grin_wl(x, y + h, z, wl, gx, gy, gz, gwl, values,
                              x_off, x_size, y_off, y_size, z_off, z_size,
                              wl_off, wl_size, val_off)
    ny_m = interp_4d_grin_wl(x, y - h, z, wl, gx, gy, gz, gwl, values,
                              x_off, x_size, y_off, y_size, z_off, z_size,
                              wl_off, wl_size, val_off)
    nz_p = interp_4d_grin_wl(x, y, z + h, wl, gx, gy, gz, gwl, values,
                              x_off, x_size, y_off, y_size, z_off, z_size,
                              wl_off, wl_size, val_off)
    nz_m = interp_4d_grin_wl(x, y, z - h, wl, gx, gy, gz, gwl, values,
                              x_off, x_size, y_off, y_size, z_off, z_size,
                              wl_off, wl_size, val_off)

    grad = wp.vec3(
        (nx_p - nx_m) / (2.0 * h),
        (ny_p - ny_m) / (2.0 * h),
        (nz_p - nz_m) / (2.0 * h),
    )
    return n_val, grad


# ==============================================================================
# GRIN 抛物线
# ==============================================================================

@wp.func
def eval_grin_parabolic(r: wp.vec3, n0: wp.float32, A: wp.float32):
    n = n0 - A * (r[0] * r[0] + r[1] * r[1])
    if n < 1.0:
        n = 1.0
    grad = wp.vec3(-2.0 * A * r[0], -2.0 * A * r[1], 0.0)
    return n, grad


# ==============================================================================
# 属性求值
# ==============================================================================

@wp.func
def eval_property(
    mat_id: wp.int32,
    prop_id: wp.int32,
    wl: wp.float32,
    cos_i: wp.float32,
    pos: wp.vec3,
    prop_kind: wp.array(dtype=wp.int32, ndim=2),
    prop_scalar: wp.array(dtype=wp.float32, ndim=2),
    prop_data_offset: wp.array(dtype=wp.int32, ndim=2),
    prop_data_size: wp.array(dtype=wp.int32, ndim=2),
    wl_grid: wp.array(dtype=wp.float32),
    wl_grid_offset: wp.array(dtype=wp.int32, ndim=2),
    wl_grid_size: wp.array(dtype=wp.int32, ndim=2),
    cos_grid: wp.array(dtype=wp.float32),
    cos_grid_offset: wp.array(dtype=wp.int32, ndim=2),
    cos_grid_size: wp.array(dtype=wp.int32, ndim=2),
    gx: wp.array(dtype=wp.float32),
    gx_offset: wp.array(dtype=wp.int32, ndim=2),
    gx_size: wp.array(dtype=wp.int32, ndim=2),
    gy: wp.array(dtype=wp.float32),
    gy_offset: wp.array(dtype=wp.int32, ndim=2),
    gy_size: wp.array(dtype=wp.int32, ndim=2),
    gz: wp.array(dtype=wp.float32),
    gz_offset: wp.array(dtype=wp.int32, ndim=2),
    gz_size: wp.array(dtype=wp.int32, ndim=2),
    prop_values: wp.array(dtype=wp.float32),
):
    kind = prop_kind[mat_id, prop_id]

    if kind == 0:
        return prop_scalar[mat_id, prop_id]

    if kind == 1:
        val_off = prop_data_offset[mat_id, prop_id]
        K = prop_data_size[mat_id, prop_id]
        g_off = wl_grid_offset[mat_id, prop_id]
        return interp_1d_full(wl, wl_grid, prop_values, g_off, K)

    if kind == 2:
        val_off = prop_data_offset[mat_id, prop_id]
        w_off = wl_grid_offset[mat_id, prop_id]
        w_sz = wl_grid_size[mat_id, prop_id]
        c_off = cos_grid_offset[mat_id, prop_id]
        c_sz = cos_grid_size[mat_id, prop_id]
        return interp_2d_full(wl, cos_i, wl_grid, cos_grid, prop_values,
                                w_off, w_sz, c_off, c_sz, val_off)

    if kind == 3:
        val_off = prop_data_offset[mat_id, prop_id]
        x_off = gx_offset[mat_id, prop_id]
        x_sz = gx_size[mat_id, prop_id]
        y_off = gy_offset[mat_id, prop_id]
        y_sz = gy_size[mat_id, prop_id]
        z_off = gz_offset[mat_id, prop_id]
        z_sz = gz_size[mat_id, prop_id]
        return interp_3d_grin(pos[0], pos[1], pos[2], gx, gy, gz,
                                prop_values,
                                x_off, x_sz, y_off, y_sz, z_off, z_sz,
                                val_off)

    if kind == 4:
        val_off = prop_data_offset[mat_id, prop_id]
        x_off = gx_offset[mat_id, prop_id]
        x_sz = gx_size[mat_id, prop_id]
        y_off = gy_offset[mat_id, prop_id]
        y_sz = gy_size[mat_id, prop_id]
        z_off = gz_offset[mat_id, prop_id]
        z_sz = gz_size[mat_id, prop_id]
        w_off = wl_grid_offset[mat_id, prop_id]
        w_sz = wl_grid_size[mat_id, prop_id]
        return interp_4d_grin_wl(pos[0], pos[1], pos[2], wl,
                                   gx, gy, gz, wl_grid, prop_values,
                                   x_off, x_sz, y_off, y_sz, z_off, z_sz,
                                   w_off, w_sz, val_off)

    return prop_scalar[mat_id, prop_id]


@wp.func
def ray_box_exit_distance(
    origin: wp.vec3,
    d: wp.vec3,
    bbox_min: wp.vec3,
    bbox_max: wp.vec3,
    default_dist: wp.float32,
) -> wp.float32:
    """Slab method: 光线与 AABB 的远交点距离（t_far）。

    如果光线完全不穿过 bbox，返回 default_dist。
    """
    eps = wp.float32(1e-12)

    # 防止除零
    safe_dx = wp.where(wp.abs(d[0]) < eps,
                       wp.where(d[0] >= 0.0, eps, -eps), d[0])
    safe_dy = wp.where(wp.abs(d[1]) < eps,
                       wp.where(d[1] >= 0.0, eps, -eps), d[1])
    safe_dz = wp.where(wp.abs(d[2]) < eps,
                       wp.where(d[2] >= 0.0, eps, -eps), d[2])
    inv_dx = 1.0 / safe_dx
    inv_dy = 1.0 / safe_dy
    inv_dz = 1.0 / safe_dz

    # slab 交点
    t1x = (bbox_min[0] - origin[0]) * inv_dx
    t2x = (bbox_max[0] - origin[0]) * inv_dx
    t1y = (bbox_min[1] - origin[1]) * inv_dy
    t2y = (bbox_max[1] - origin[1]) * inv_dy
    t1z = (bbox_min[2] - origin[2]) * inv_dz
    t2z = (bbox_max[2] - origin[2]) * inv_dz

    t_near_x = wp.min(t1x, t2x)
    t_near_y = wp.min(t1y, t2y)
    t_near_z = wp.min(t1z, t2z)
    t_far_x = wp.max(t1x, t2x)
    t_far_y = wp.max(t1y, t2y)
    t_far_z = wp.max(t1z, t2z)

    t_near = wp.max(wp.max(t_near_x, t_near_y), t_near_z)
    t_far = wp.min(wp.min(t_far_x, t_far_y), t_far_z)

    # 有效性：t_far >= max(t_near, 0)
    valid = (t_far >= wp.max(t_near, 0.0)) and (t_far >= 0.0)
    if valid:
        return t_far
    return default_dist

@wp.func
def eval_grin_n_and_grad(
    mat_id: wp.int32,
    pos: wp.vec3,
    wl: wp.float32,
    grin_kind: wp.array(dtype=wp.int32),
    grin_n0: wp.array(dtype=wp.float32),
    grin_A: wp.array(dtype=wp.float32),
    prop_data_offset: wp.array(dtype=wp.int32, ndim=2),
    prop_data_size: wp.array(dtype=wp.int32, ndim=2),
    gx: wp.array(dtype=wp.float32),
    gx_offset: wp.array(dtype=wp.int32, ndim=2),
    gx_size: wp.array(dtype=wp.int32, ndim=2),
    gy: wp.array(dtype=wp.float32),
    gy_offset: wp.array(dtype=wp.int32, ndim=2),
    gy_size: wp.array(dtype=wp.int32, ndim=2),
    gz: wp.array(dtype=wp.float32),
    gz_offset: wp.array(dtype=wp.int32, ndim=2),
    gz_size: wp.array(dtype=wp.int32, ndim=2),
    wl_grid: wp.array(dtype=wp.float32),
    wl_grid_offset: wp.array(dtype=wp.int32, ndim=2),
    wl_grid_size: wp.array(dtype=wp.int32, ndim=2),
    prop_values: wp.array(dtype=wp.float32),
):
    gkind = grin_kind[mat_id]

    if gkind == 1:
        return eval_grin_parabolic(pos, grin_n0[mat_id], grin_A[mat_id])

    if gkind == 3:
        val_off = prop_data_offset[mat_id, 0]
        x_off = gx_offset[mat_id, 0]
        x_sz = gx_size[mat_id, 0]
        y_off = gy_offset[mat_id, 0]
        y_sz = gy_size[mat_id, 0]
        z_off = gz_offset[mat_id, 0]
        z_sz = gz_size[mat_id, 0]
        return interp_3d_grin_and_grad(pos[0], pos[1], pos[2],
                                         gx, gy, gz, prop_values,
                                         x_off, x_sz, y_off, y_sz,
                                         z_off, z_sz, val_off)

    if gkind == 4:
        val_off = prop_data_offset[mat_id, 0]
        x_off = gx_offset[mat_id, 0]
        x_sz = gx_size[mat_id, 0]
        y_off = gy_offset[mat_id, 0]
        y_sz = gy_size[mat_id, 0]
        z_off = gz_offset[mat_id, 0]
        z_sz = gz_size[mat_id, 0]
        w_off = wl_grid_offset[mat_id, 0]
        w_sz = wl_grid_size[mat_id, 0]
        return interp_4d_grin_wl_and_grad(pos[0], pos[1], pos[2], wl,
                                            gx, gy, gz, wl_grid, prop_values,
                                            x_off, x_sz, y_off, y_sz,
                                            z_off, z_sz, w_off, w_sz,
                                            val_off)

    return wp.float32(1.0), wp.vec3(0.0, 0.0, 0.0)


# ==============================================================================
# 主内核
# ==============================================================================

@wp.kernel(enable_backward=False)
def trace_kernel(
    mesh_id: wp.uint64,
    tri_v0: wp.array(dtype=wp.vec3),
    tri_v1: wp.array(dtype=wp.vec3),
    tri_v2: wp.array(dtype=wp.vec3),
    face_materials: wp.array(dtype=wp.int32, ndim=2),
    prop_kind: wp.array(dtype=wp.int32, ndim=2),
    prop_scalar: wp.array(dtype=wp.float32, ndim=2),
    prop_data_offset: wp.array(dtype=wp.int32, ndim=2),
    prop_data_size: wp.array(dtype=wp.int32, ndim=2),
    prop_values: wp.array(dtype=wp.float32),
    wl_grid: wp.array(dtype=wp.float32),
    wl_grid_offset: wp.array(dtype=wp.int32, ndim=2),
    wl_grid_size: wp.array(dtype=wp.int32, ndim=2),
    cos_grid: wp.array(dtype=wp.float32),
    cos_grid_offset: wp.array(dtype=wp.int32, ndim=2),
    cos_grid_size: wp.array(dtype=wp.int32, ndim=2),
    gx: wp.array(dtype=wp.float32),
    gx_offset: wp.array(dtype=wp.int32, ndim=2),
    gx_size: wp.array(dtype=wp.int32, ndim=2),
    gy: wp.array(dtype=wp.float32),
    gy_offset: wp.array(dtype=wp.int32, ndim=2),
    gy_size: wp.array(dtype=wp.int32, ndim=2),
    gz: wp.array(dtype=wp.float32),
    gz_offset: wp.array(dtype=wp.int32, ndim=2),
    gz_size: wp.array(dtype=wp.int32, ndim=2),
    grin_kind: wp.array(dtype=wp.int32),
    grin_n0: wp.array(dtype=wp.float32),
    grin_A: wp.array(dtype=wp.float32),
    ray_origins: wp.array(dtype=wp.vec3),
    ray_dirs: wp.array(dtype=wp.vec3),
    ray_stokes: wp.array(dtype=wp.vec4),
    ray_wavelengths: wp.array(dtype=wp.float32),
    ray_medium_ids: wp.array(dtype=wp.int32),
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
    max_steps: wp.int32,
    eps: wp.float32,
    far: wp.float32,
    far_escape: wp.float32,
    scene_bbox_min: wp.vec3,
    scene_bbox_max: wp.vec3,
    seed: wp.int32,
    surface_model_flag: wp.int32,
    ds_init: wp.float32,
    num_media: wp.int32,
):
    tid = wp.tid()

    origin = ray_origins[tid]
    d = ray_dirs[tid]
    S0 = ray_stokes[tid][0]
    S1 = ray_stokes[tid][1]
    S2 = ray_stokes[tid][2]
    S3 = ray_stokes[tid][3]
    wl = ray_wavelengths[tid]
    weight = wp.float32(1.0)
    active = wp.int32(1)
    medium_id = ray_medium_ids[tid]
    prev_face = wp.int32(-1)
    s_ref = choose_perpendicular(d)
    ds = ds_init

    rng = wp.rand_init(seed, tid)

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
        # 求交 + 推开自相交（不消耗 step）
        q = wp.mesh_query_ray(mesh_id, origin, d, far)
        skip_guard = wp.int32(0)
        while skip_guard < 8:
            if not q.result:
                break
            if q.t > eps and q.face != prev_face:
                break
            origin = origin + d * (q.t + eps)
            q = wp.mesh_query_ray(mesh_id, origin, d, far)
            skip_guard = skip_guard + 1

        gkind = wp.int32(0)
        if medium_id >= 0:
            if medium_id < num_media:
                gkind = grin_kind[medium_id]
        g_active = (gkind > 0)

        if g_active and ((not q.result) or q.t > ds):
            # ========================================================
            # GRIN RK4 步进
            # ========================================================
            n_curr, grad_curr = eval_grin_n_and_grad(
                medium_id, origin, wl, grin_kind, grin_n0, grin_A,
                prop_data_offset, prop_data_size,
                gx, gx_offset, gx_size,
                gy, gy_offset, gy_size,
                gz, gz_offset, gz_size,
                wl_grid, wl_grid_offset, wl_grid_size,
                prop_values)

            p = n_curr * d
            dr1 = p / n_curr
            dp1 = grad_curr

            r2 = origin + 0.5 * ds * dr1
            p2 = p + 0.5 * ds * dp1
            n2, g2 = eval_grin_n_and_grad(
                medium_id, r2, wl, grin_kind, grin_n0, grin_A,
                prop_data_offset, prop_data_size,
                gx, gx_offset, gx_size,
                gy, gy_offset, gy_size,
                gz, gz_offset, gz_size,
                wl_grid, wl_grid_offset, wl_grid_size,
                prop_values)
            dr2 = p2 / n2
            dp2 = g2

            r3 = origin + 0.5 * ds * dr2
            p3 = p + 0.5 * ds * dp2
            n3, g3 = eval_grin_n_and_grad(
                medium_id, r3, wl, grin_kind, grin_n0, grin_A,
                prop_data_offset, prop_data_size,
                gx, gx_offset, gx_size,
                gy, gy_offset, gy_size,
                gz, gz_offset, gz_size,
                wl_grid, wl_grid_offset, wl_grid_size,
                prop_values)
            dr3 = p3 / n3
            dp3 = g3

            r4 = origin + ds * dr3
            p4 = p + ds * dp3
            n4, g4 = eval_grin_n_and_grad(
                medium_id, r4, wl, grin_kind, grin_n0, grin_A,
                prop_data_offset, prop_data_size,
                gx, gx_offset, gx_size,
                gy, gy_offset, gy_size,
                gz, gz_offset, gz_size,
                wl_grid, wl_grid_offset, wl_grid_size,
                prop_values)
            dr4 = p4 / n4
            dp4 = g4

            new_origin = origin + (ds / 6.0) * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4)
            p_next = p + (ds / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
            new_n, _ = eval_grin_n_and_grad(
                medium_id, new_origin, wl, grin_kind, grin_n0, grin_A,
                prop_data_offset, prop_data_size,
                gx, gx_offset, gx_size,
                gy, gy_offset, gy_size,
                gz, gz_offset, gz_size,
                wl_grid, wl_grid_offset, wl_grid_size,
                prop_values)
            new_d = wp.normalize(p_next / new_n)

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
            # ========================================================
            # 表面交互
            # ========================================================
            cur_refl = wp.float32(0.0)
            cur_trans = wp.float32(0.0)
            cur_abs = wp.float32(0.0)
            cur_n1 = wp.float32(1.0)
            cur_n2 = wp.float32(1.0)

            if not q.result:
                # ★ 逃逸光线终止于场景 bbox 边界
                default_dist = 2.0 * far_escape / 100.0  # 用 scene_diag 的 2 倍
                # 注意：far_escape 在 __init__ 里设为 100*scene_diag
                # 所以 far_escape/100 = scene_diag
                t_exit = ray_box_exit_distance(
                    origin, d, scene_bbox_min, scene_bbox_max,
                    default_dist)
                # 光线至少走一点点，避免 t_exit = 0
                t_exit = wp.max(t_exit, wp.float32(0.1))
                origin = origin + d * t_exit
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

                gkind_1 = wp.int32(0)
                if n1_id >= 0:
                    if n1_id < num_media:
                        gkind_1 = grin_kind[n1_id]
                if gkind_1 > 0:
                    n1, _ = eval_grin_n_and_grad(
                        n1_id, origin, wl, grin_kind, grin_n0, grin_A,
                        prop_data_offset, prop_data_size,
                        gx, gx_offset, gx_size,
                        gy, gy_offset, gy_size,
                        gz, gz_offset, gz_size,
                        wl_grid, wl_grid_offset, wl_grid_size,
                        prop_values)
                else:
                    n1 = eval_property(
                        n1_id, 0, wl, cos_i, origin,
                        prop_kind, prop_scalar,
                        prop_data_offset, prop_data_size,
                        wl_grid, wl_grid_offset, wl_grid_size,
                        cos_grid, cos_grid_offset, cos_grid_size,
                        gx, gx_offset, gx_size,
                        gy, gy_offset, gy_size,
                        gz, gz_offset, gz_size,
                        prop_values)

                gkind_2 = wp.int32(0)
                if n2_id >= 0:
                    if n2_id < num_media:
                        gkind_2 = grin_kind[n2_id]
                if gkind_2 > 0:
                    n2, _ = eval_grin_n_and_grad(
                        n2_id, origin, wl, grin_kind, grin_n0, grin_A,
                        prop_data_offset, prop_data_size,
                        gx, gx_offset, gx_size,
                        gy, gy_offset, gy_size,
                        gz, gz_offset, gz_size,
                        wl_grid, wl_grid_offset, wl_grid_size,
                        prop_values)
                else:
                    n2 = eval_property(
                        n2_id, 0, wl, cos_i, origin,
                        prop_kind, prop_scalar,
                        prop_data_offset, prop_data_size,
                        wl_grid, wl_grid_offset, wl_grid_size,
                        cos_grid, cos_grid_offset, cos_grid_size,
                        gx, gx_offset, gx_size,
                        gy, gy_offset, gy_size,
                        gz, gz_offset, gz_size,
                        prop_values)

                if n1 < 1.0:
                    n1 = 1.0
                if n2 < 1.0:
                    n2 = 1.0

                custom_refl = eval_property(
                    n2_id, 1, wl, cos_i, origin,
                    prop_kind, prop_scalar,
                    prop_data_offset, prop_data_size,
                    wl_grid, wl_grid_offset, wl_grid_size,
                    cos_grid, cos_grid_offset, cos_grid_size,
                    gx, gx_offset, gx_size,
                    gy, gy_offset, gy_size,
                    gz, gz_offset, gz_size,
                    prop_values)
                custom_diff = eval_property(
                    n2_id, 2, wl, cos_i, origin,
                    prop_kind, prop_scalar,
                    prop_data_offset, prop_data_size,
                    wl_grid, wl_grid_offset, wl_grid_size,
                    cos_grid, cos_grid_offset, cos_grid_size,
                    gx, gx_offset, gx_size,
                    gy, gy_offset, gy_size,
                    gz, gz_offset, gz_size,
                    prop_values)
                custom_abs = eval_property(
                    n2_id, 3, wl, cos_i, origin,
                    prop_kind, prop_scalar,
                    prop_data_offset, prop_data_size,
                    wl_grid, wl_grid_offset, wl_grid_size,
                    cos_grid, cos_grid_offset, cos_grid_size,
                    gx, gx_offset, gx_size,
                    gy, gy_offset, gy_size,
                    gz, gz_offset, gz_size,
                    prop_values)

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


# ==============================================================================
# LUT 构建器
# ==============================================================================

PROP_N = 0
PROP_R = 1
PROP_D = 2
PROP_A = 3


class _LUTBuilder:
    def __init__(self, num_mat):
        self.num_mat = num_mat
        self.prop_kind = np.zeros((num_mat, 4), dtype=np.int32)
        self.prop_scalar = np.zeros((num_mat, 4), dtype=np.float32)
        self.prop_data_offset = np.zeros((num_mat, 4), dtype=np.int32)
        self.prop_data_size = np.zeros((num_mat, 4), dtype=np.int32)

        self.wl_grid_parts = []
        self.wl_grid_offset = np.zeros((num_mat, 4), dtype=np.int32)
        self.wl_grid_size = np.zeros((num_mat, 4), dtype=np.int32)
        self.wl_grid_len = 0

        self.cos_grid_parts = []
        self.cos_grid_offset = np.zeros((num_mat, 4), dtype=np.int32)
        self.cos_grid_size = np.zeros((num_mat, 4), dtype=np.int32)
        self.cos_grid_len = 0

        self.gx_parts = []
        self.gx_offset = np.zeros((num_mat, 4), dtype=np.int32)
        self.gx_size = np.zeros((num_mat, 4), dtype=np.int32)
        self.gx_len = 0

        self.gy_parts = []
        self.gy_offset = np.zeros((num_mat, 4), dtype=np.int32)
        self.gy_size = np.zeros((num_mat, 4), dtype=np.int32)
        self.gy_len = 0

        self.gz_parts = []
        self.gz_offset = np.zeros((num_mat, 4), dtype=np.int32)
        self.gz_size = np.zeros((num_mat, 4), dtype=np.int32)
        self.gz_len = 0

        self.values_parts = []
        self.values_len = 0

    def _add_wl_grid(self, mat_id, prop_id, wl_nodes):
        arr = np.asarray(wl_nodes, dtype=np.float32)
        self.wl_grid_offset[mat_id, prop_id] = self.wl_grid_len
        self.wl_grid_size[mat_id, prop_id] = len(arr)
        self.wl_grid_parts.append(arr)
        self.wl_grid_len += len(arr)

    def _add_cos_grid(self, mat_id, prop_id, cos_nodes):
        arr = np.asarray(cos_nodes, dtype=np.float32)
        self.cos_grid_offset[mat_id, prop_id] = self.cos_grid_len
        self.cos_grid_size[mat_id, prop_id] = len(arr)
        self.cos_grid_parts.append(arr)
        self.cos_grid_len += len(arr)

    def _add_gx(self, mat_id, prop_id, x_nodes):
        arr = np.asarray(x_nodes, dtype=np.float32)
        self.gx_offset[mat_id, prop_id] = self.gx_len
        self.gx_size[mat_id, prop_id] = len(arr)
        self.gx_parts.append(arr)
        self.gx_len += len(arr)

    def _add_gy(self, mat_id, prop_id, y_nodes):
        arr = np.asarray(y_nodes, dtype=np.float32)
        self.gy_offset[mat_id, prop_id] = self.gy_len
        self.gy_size[mat_id, prop_id] = len(arr)
        self.gy_parts.append(arr)
        self.gy_len += len(arr)

    def _add_gz(self, mat_id, prop_id, z_nodes):
        arr = np.asarray(z_nodes, dtype=np.float32)
        self.gz_offset[mat_id, prop_id] = self.gz_len
        self.gz_size[mat_id, prop_id] = len(arr)
        self.gz_parts.append(arr)
        self.gz_len += len(arr)

    def _add_values(self, mat_id, prop_id, vals):
        arr = np.asarray(vals, dtype=np.float32).flatten()
        self.prop_data_offset[mat_id, prop_id] = self.values_len
        self.prop_data_size[mat_id, prop_id] = len(arr)
        self.values_parts.append(arr)
        self.values_len += len(arr)

    def add_scalar(self, mat_id, prop_id, value):
        self.prop_kind[mat_id, prop_id] = 0
        self.prop_scalar[mat_id, prop_id] = float(value)

    def add_wl_lut(self, mat_id, prop_id, wl_nodes, values):
        self.prop_kind[mat_id, prop_id] = 1
        self._add_wl_grid(mat_id, prop_id, wl_nodes)
        self._add_values(mat_id, prop_id, values)
        vals_arr = np.asarray(values).flatten()
        self.prop_scalar[mat_id, prop_id] = float(vals_arr[len(vals_arr) // 2])

    def add_2d_lut(self, mat_id, prop_id, wl_nodes, cos_nodes, values_2d):
        self.prop_kind[mat_id, prop_id] = 2
        self._add_wl_grid(mat_id, prop_id, wl_nodes)
        self._add_cos_grid(mat_id, prop_id, cos_nodes)
        self._add_values(mat_id, prop_id, values_2d)

    def add_grin_3d(self, mat_id, x_nodes, y_nodes, z_nodes, values_3d):
        self.prop_kind[mat_id, PROP_N] = 3
        self._add_gx(mat_id, PROP_N, x_nodes)
        self._add_gy(mat_id, PROP_N, y_nodes)
        self._add_gz(mat_id, PROP_N, z_nodes)
        arr = np.asarray(values_3d, dtype=np.float32)
        self._add_values(mat_id, PROP_N, arr)
        self.prop_scalar[mat_id, PROP_N] = float(arr.flatten()[0])

    def add_grin_4d(self, mat_id, x_nodes, y_nodes, z_nodes, wl_nodes, values_4d):
        self.prop_kind[mat_id, PROP_N] = 4
        self._add_gx(mat_id, PROP_N, x_nodes)
        self._add_gy(mat_id, PROP_N, y_nodes)
        self._add_gz(mat_id, PROP_N, z_nodes)
        self._add_wl_grid(mat_id, PROP_N, wl_nodes)
        arr = np.asarray(values_4d, dtype=np.float32)
        self._add_values(mat_id, PROP_N, arr)
        self.prop_scalar[mat_id, PROP_N] = float(arr.flatten()[0])

    def build(self, device):
        def _cat(parts, dtype=np.float32):
            if not parts:
                return np.zeros(0, dtype=dtype)
            return np.concatenate(parts).astype(dtype)

        return {
            "prop_kind": wp.array(self.prop_kind, dtype=wp.int32, ndim=2, device=device),
            "prop_scalar": wp.array(self.prop_scalar, dtype=wp.float32, ndim=2, device=device),
            "prop_data_offset": wp.array(self.prop_data_offset, dtype=wp.int32, ndim=2, device=device),
            "prop_data_size": wp.array(self.prop_data_size, dtype=wp.int32, ndim=2, device=device),
            "prop_values": wp.array(_cat(self.values_parts), dtype=wp.float32, device=device),
            "wl_grid": wp.array(_cat(self.wl_grid_parts), dtype=wp.float32, device=device),
            "wl_grid_offset": wp.array(self.wl_grid_offset, dtype=wp.int32, ndim=2, device=device),
            "wl_grid_size": wp.array(self.wl_grid_size, dtype=wp.int32, ndim=2, device=device),
            "cos_grid": wp.array(_cat(self.cos_grid_parts), dtype=wp.float32, device=device),
            "cos_grid_offset": wp.array(self.cos_grid_offset, dtype=wp.int32, ndim=2, device=device),
            "cos_grid_size": wp.array(self.cos_grid_size, dtype=wp.int32, ndim=2, device=device),
            "gx": wp.array(_cat(self.gx_parts), dtype=wp.float32, device=device),
            "gx_offset": wp.array(self.gx_offset, dtype=wp.int32, ndim=2, device=device),
            "gx_size": wp.array(self.gx_size, dtype=wp.int32, ndim=2, device=device),
            "gy": wp.array(_cat(self.gy_parts), dtype=wp.float32, device=device),
            "gy_offset": wp.array(self.gy_offset, dtype=wp.int32, ndim=2, device=device),
            "gy_size": wp.array(self.gy_size, dtype=wp.int32, ndim=2, device=device),
            "gz": wp.array(_cat(self.gz_parts), dtype=wp.float32, device=device),
            "gz_offset": wp.array(self.gz_offset, dtype=wp.int32, ndim=2, device=device),
            "gz_size": wp.array(self.gz_size, dtype=wp.int32, ndim=2, device=device),
        }


# ==============================================================================
# Python 包装类
# ==============================================================================

class WarpTracer:
    """全 Warp 内核 tracer。"""

    def __init__(self, vertices, faces, face_materials, materials_cfg, device,
                 grin_callable=None):
        # ★ 验证材质配置（复用 PyTorch tracer 的验证器）
        _validate_materials_cfg(materials_cfg, face_materials)

        # ★ 检查 warp 是否可用
        if not hasattr(wp, "init"):
            raise BackendError(
                "Warp not available. Install with: pip install warp-lang")

        self.device = device
        self.warp_device = "cuda:0" if str(device).startswith("cuda") else "cpu"
        self.grin_callable = grin_callable

        wp.init()

        verts_np = vertices.detach().cpu().numpy().astype(np.float32)
        faces_np = faces.detach().cpu().numpy().astype(np.int32).flatten()

        self.mesh = wp.Mesh(
            points=wp.array(verts_np, dtype=wp.vec3, device=self.warp_device),
            indices=wp.array(faces_np, dtype=wp.int32, device=self.warp_device),
        )
        self.mesh_id = self.mesh.id
        # ★ 场景对角线（用于逃逸光线的远场延伸）
        # ★ 场景 bbox —— 逃逸光线终止于 bbox 边界
        v_min = verts_np.min(axis=0)
        v_max = verts_np.max(axis=0)
        diag = float(np.linalg.norm(v_max - v_min))
        self.scene_diag = diag if diag > 1e-6 else 1.0
        # far_escape 保留（用于传给 kernel 作为兜底），但实际用 bbox 边界
        self.far_escape = max(100.0 * self.scene_diag, 1e5)

        # bbox 加 5% padding（避免光线终点正好在边界上）
        pad = 0.05 * (v_max - v_min)
        self.scene_bbox_min_np = (v_min - pad).astype(np.float32)
        self.scene_bbox_max_np = (v_max + pad).astype(np.float32)

        faces_long = faces.detach().cpu().numpy().astype(np.int64)
        tris = verts_np[faces_long]
        self.tri_v0 = wp.array(np.ascontiguousarray(tris[:, 0]),
                                dtype=wp.vec3, device=self.warp_device)
        self.tri_v1 = wp.array(np.ascontiguousarray(tris[:, 1]),
                                dtype=wp.vec3, device=self.warp_device)
        self.tri_v2 = wp.array(np.ascontiguousarray(tris[:, 2]),
                                dtype=wp.vec3, device=self.warp_device)

        fm_np = face_materials.detach().cpu().numpy().astype(np.int32)
        self.face_materials = wp.array(fm_np, dtype=wp.int32, ndim=2,
                                         device=self.warp_device)

        mesh_min = verts_np.min(axis=0)
        mesh_max = verts_np.max(axis=0)

        max_mat = max(materials_cfg.keys()) + 1
        self.num_media = max_mat
        lut = _LUTBuilder(max_mat)

        grin_kind_arr = np.zeros(max_mat, dtype=np.int32)
        grin_n0_arr = np.ones(max_mat, dtype=np.float32)
        grin_A_arr = np.zeros(max_mat, dtype=np.float32)

        for k, cfg in materials_cfg.items():
            n_cfg = cfg.get("n", 1.0)
            self._add_property(lut, k, PROP_N, n_cfg)
            self._add_property(lut, k, PROP_R, cfg.get("reflectivity", 0.0))
            self._add_property(lut, k, PROP_D, cfg.get("diffuse", 0.0))
            self._add_property(lut, k, PROP_A, cfg.get("absorptivity", 0.0))

            gf = cfg.get("grin_field", None)
            if gf is not None:
                if isinstance(gf, dict):
                    gtype = gf.get("type", "")
                    if gtype == "parabolic":
                        grin_kind_arr[k] = 1
                        n0 = float(gf.get("n0", 1.5))
                        A = float(gf.get("A", 0.0))
                        grin_n0_arr[k] = n0
                        grin_A_arr[k] = A
                        lut.prop_scalar[k, PROP_N] = n0
                    elif gtype == "grid":
                        x = np.asarray(gf["x"], dtype=np.float32)
                        y = np.asarray(gf["y"], dtype=np.float32)
                        z = np.asarray(gf["z"], dtype=np.float32)
                        values = np.asarray(gf["values"], dtype=np.float32)
                        assert values.shape == (len(x), len(y), len(z))
                        lut.add_grin_3d(k, x, y, z, values)
                        grin_kind_arr[k] = 3
                    elif gtype == "grid_4d":
                        x = np.asarray(gf["x"], dtype=np.float32)
                        y = np.asarray(gf["y"], dtype=np.float32)
                        z = np.asarray(gf["z"], dtype=np.float32)
                        wl = np.asarray(gf["wavelengths_nm"], dtype=np.float32)
                        values = np.asarray(gf["values"], dtype=np.float32)
                        assert values.shape == (len(x), len(y), len(z), len(wl))
                        lut.add_grin_4d(k, x, y, z, wl, values)
                        grin_kind_arr[k] = 4
                    elif gtype == "callable":
                        if grin_callable is None:
                            raise ValueError(
                                "Material {} uses grin_field={{'type': 'callable'}} "
                                "but grin_callable is None".format(k))
                        # ★ 采样 callable 为 3D grid
                        bounds = gf.get("bounds", None)
                        resolution = tuple(gf.get("resolution", (32, 32, 32)))
                        x_nodes, y_nodes, z_nodes, values = \
                            self._sample_callable_to_grid(
                                grin_callable, bounds, resolution,
                                (mesh_min, mesh_max))
                        lut.add_grin_3d(k, x_nodes, y_nodes, z_nodes, values)
                        grin_kind_arr[k] = 3
                    else:
                        raise NotImplementedError(
                            "Unsupported grin type: {}".format(gtype))
                else:
                    raise NotImplementedError("grin_field must be dict")

        lut_arrays = lut.build(self.warp_device)
        self.prop_kind = lut_arrays["prop_kind"]
        self.prop_scalar = lut_arrays["prop_scalar"]
        self.prop_data_offset = lut_arrays["prop_data_offset"]
        self.prop_data_size = lut_arrays["prop_data_size"]
        self.prop_values = lut_arrays["prop_values"]
        self.wl_grid = lut_arrays["wl_grid"]
        self.wl_grid_offset = lut_arrays["wl_grid_offset"]
        self.wl_grid_size = lut_arrays["wl_grid_size"]
        self.cos_grid = lut_arrays["cos_grid"]
        self.cos_grid_offset = lut_arrays["cos_grid_offset"]
        self.cos_grid_size = lut_arrays["cos_grid_size"]
        self.gx = lut_arrays["gx"]
        self.gx_offset = lut_arrays["gx_offset"]
        self.gx_size = lut_arrays["gx_size"]
        self.gy = lut_arrays["gy"]
        self.gy_offset = lut_arrays["gy_offset"]
        self.gy_size = lut_arrays["gy_size"]
        self.gz = lut_arrays["gz"]
        self.gz_offset = lut_arrays["gz_offset"]
        self.gz_size = lut_arrays["gz_size"]

        self.grin_kind = wp.array(grin_kind_arr, dtype=wp.int32,
                                    device=self.warp_device)
        self.grin_n0 = wp.array(grin_n0_arr, dtype=wp.float32,
                                  device=self.warp_device)
        self.grin_A = wp.array(grin_A_arr, dtype=wp.float32,
                                 device=self.warp_device)

    @staticmethod
    def _sample_callable_to_grid(func, bounds, resolution, mesh_bbox):
        """
        把 Python callable 采样成 3D grid。

        Args:
            func: Python 函数 f(x, y, z) -> n，支持标量或 numpy 数组。
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax) 或 None。
            resolution: (nx, ny, nz) 采样点数。
            mesh_bbox: (mesh_min, mesh_max) numpy 数组。

        Returns:
            (x_nodes, y_nodes, z_nodes, values)
        """
        nx, ny, nz = resolution

        if bounds is None:
            mesh_min, mesh_max = mesh_bbox
            span = mesh_max - mesh_min
            margin = 0.2 * span
            xmin = float(mesh_min[0] - margin[0])
            xmax = float(mesh_max[0] + margin[0])
            ymin = float(mesh_min[1] - margin[1])
            ymax = float(mesh_max[1] + margin[1])
            zmin = float(mesh_min[2] - margin[2])
            zmax = float(mesh_max[2] + margin[2])
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = bounds

        x_nodes = np.linspace(xmin, xmax, nx).astype(np.float32)
        y_nodes = np.linspace(ymin, ymax, ny).astype(np.float32)
        z_nodes = np.linspace(zmin, zmax, nz).astype(np.float32)

        X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
        try:
            values = func(X, Y, Z)
        except (TypeError, ValueError):
            values = np.zeros_like(X)
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        values[i, j, k] = float(func(X[i, j, k],
                                                     Y[i, j, k],
                                                     Z[i, j, k]))

        values = np.asarray(values, dtype=np.float32)
        if values.shape != (nx, ny, nz):
            raise ValueError(
                "Callable returned shape {} but expected ({}, {}, {})".format(
                    values.shape, nx, ny, nz))

        return x_nodes, y_nodes, z_nodes, values

    @staticmethod
    def _add_property(lut, mat_id, prop_id, cfg):
        if isinstance(cfg, (int, float)):
            lut.add_scalar(mat_id, prop_id, float(cfg))
            return

        if not isinstance(cfg, dict):
            raise NotImplementedError(
                "Property config must be scalar or dict, got {}".format(type(cfg)))

        if "wavelengths_nm" in cfg and "cos_theta" in cfg and "values" in cfg:
            lut.add_2d_lut(mat_id, prop_id,
                            cfg["wavelengths_nm"], cfg["cos_theta"], cfg["values"])
            return

        if "wavelengths_nm" in cfg and "values" in cfg:
            lut.add_wl_lut(mat_id, prop_id, cfg["wavelengths_nm"], cfg["values"])
            return

        raise NotImplementedError("Unrecognized property config: {}".format(cfg))

    def trace(self, ray_dict, max_steps=50, ds_init=0.02, ds_min=1e-3,
              ds_max=0.1, rk_tol=1e-4, fresnel=True, verbose=False,
              surface_model=None, rr_threshold=0.0, seed=42):
        """追踪光线。

        Args:
            ray_dict: generate_rays() 的输出。
                可选键 "medium_ids": [N] long tensor，每条光线的初始介质 ID。
            ...
        """

        _validate_ray_dict(ray_dict)

        if surface_model is None:
            surface_model = "coated" if fresnel else "custom"

        if surface_model is None:
            surface_model = "coated" if fresnel else "custom"
        if surface_model not in ("fresnel", "custom", "coated"):
            raise ValueError(
                "surface_model must be 'fresnel', 'custom', or 'coated', "
                "got '{}'".format(surface_model))
        model_map = {"fresnel": 0, "custom": 1, "coated": 2}
        surface_model_flag = model_map[surface_model]

        origins = ray_dict["origins"].to(self.device).contiguous().float()
        dirs = ray_dict["dirs"].to(self.device).contiguous().float()
        stokes = ray_dict["stokes"].to(self.device).contiguous().float()
        wavelengths = ray_dict["wavelengths"].to(self.device).contiguous().float().squeeze(-1)

        # ★ N 必须在 medium_ids 处理之前定义
        N = origins.shape[0]
        S = max_steps + 1

        # ★ 支持 ray_dict["medium_ids"]：每条光线的初始介质
        if "medium_ids" in ray_dict:
            mid = ray_dict["medium_ids"]
            if isinstance(mid, int):
                medium_ids_t = torch.full((N,), int(mid), dtype=torch.int32,
                                          device=self.device)
            else:
                medium_ids_t = mid.to(dtype=torch.int32, device=self.device).contiguous()
                if medium_ids_t.shape[0] != N:
                    raise ValueError(
                        "ray_dict['medium_ids'] length {} "
                        "does not match number of rays {}".format(
                            medium_ids_t.shape[0], N))
        else:
            medium_ids_t = torch.zeros(N, dtype=torch.int32, device=self.device)
        wp_medium_ids = wp.from_torch(medium_ids_t, dtype=wp.int32)

        wp_origins = wp.from_torch(origins, dtype=wp.vec3)
        wp_dirs = wp.from_torch(dirs, dtype=wp.vec3)
        wp_stokes = wp.from_torch(stokes, dtype=wp.vec4)
        wp_wl = wp.from_torch(wavelengths, dtype=wp.float32)

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

        wp.launch(
            trace_kernel,
            dim=N,
            inputs=[
                self.mesh_id,
                self.tri_v0, self.tri_v1, self.tri_v2,
                self.face_materials,
                self.prop_kind, self.prop_scalar,
                self.prop_data_offset, self.prop_data_size,
                self.prop_values,
                self.wl_grid, self.wl_grid_offset, self.wl_grid_size,
                self.cos_grid, self.cos_grid_offset, self.cos_grid_size,
                self.gx, self.gx_offset, self.gx_size,
                self.gy, self.gy_offset, self.gy_size,
                self.gz, self.gz_offset, self.gz_size,
                self.grin_kind, self.grin_n0, self.grin_A,
                wp_origins, wp_dirs, wp_stokes, wp_wl,
                wp_medium_ids,
                wp_pos, wp_stokes_hist, wp_energy, wp_active,
                wp_refl, wp_trans, wp_abs, wp_n1, wp_n2, wp_dop,
                max_steps,
                np.float32(1e-4),
                np.float32(1e10),
                np.float32(self.far_escape),
                wp.vec3(*self.scene_bbox_min_np.tolist()),
                wp.vec3(*self.scene_bbox_max_np.tolist()),
                int(seed),
                int(surface_model_flag),
                np.float32(ds_init),
                int(self.num_media),
            ],
            device=self.warp_device,
        )
        wp.synchronize()

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