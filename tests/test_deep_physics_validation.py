"""
tests/test_deep_physics_validation.py
======================================
深度物理验证——7 个测试全部与解析解或已发表参考值对比。

覆盖：
    1. 圆偏振输入 → 椭圆偏振输出（Jones 矩阵）
    2. Fresnel 透射系数 t_s / t_p（通过 45° 偏振验证）
    3. 多次反射的偏振累积（正入射 s/p 符号约定）
    4. BK7 玻璃的 d-line 参考折射率（1.5168 @ 587.6nm）
    5. Sellmeier 色散曲线（多个波长点对比）
    6. Brewster 角偏振纯度（R_p = 0 的物理后果）
    7. 色散 + 偏振 + 能量守恒的组合场景

参考数据源：
    - Schott Glass catalog (N-BK7 Sellmeier coefficients)
    - Born & Wolf, "Principles of Optics" (7th ed)
    - Hecht, "Optics" (5th ed)

Run:
    pytest tests/test_deep_physics_validation.py -v -s
"""
import math
import numpy as np
import pytest
import torch

from georay import AdvancedRayTracerPBRTGPU, generate_rays


# ==============================================================================
# 已发表的参考数据
# ==============================================================================

def bk7_sellmeier(wl_nm: float) -> float:
    """N-BK7 玻璃在波长 λ (nm) 下的折射率（Schott 官方 Sellmeier 系数）。

    参考：Schott Glass catalog, N-BK7 datasheet
    """
    wl_um2 = (wl_nm * 1e-3) ** 2
    B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
    C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653
    n_sq = (1.0
            + B1 * wl_um2 / (wl_um2 - C1)
            + B2 * wl_um2 / (wl_um2 - C2)
            + B3 * wl_um2 / (wl_um2 - C3))
    return math.sqrt(n_sq)


# N-BK7 在几个标准谱线处的参考折射率（Schott catalog）
BK7_REFERENCE = {
    486.1: 1.52238,   # F-line (H)
    587.6: 1.51680,   # d-line (He)
    656.3: 1.51432,   # C-line (H)
}


# ==============================================================================
# 几何与 Fresnel 辅助
# ==============================================================================

def make_flat_interface(device, hs=10.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def make_two_parallel_mirrors(device, hs=5.0, z0=0.0, z1=1.0):
    verts = torch.tensor([
        [-hs, -hs, z0], [hs, -hs, z0], [hs, hs, z0], [-hs, hs, z0],
        [-hs, -hs, z1], [hs, -hs, z1], [hs, hs, z1], [-hs, hs, z1],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)
    return verts, faces


def fresnel_rs_rp(n1, n2, theta_i):
    sin_t = n1 * math.sin(theta_i) / n2
    if sin_t >= 1.0:
        return 1.0, 1.0, 0.0
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return r_s, r_p, cos_t


def fresnel_ts_tp(n1, n2, theta_i):
    sin_t = n1 * math.sin(theta_i) / n2
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    t_s = 2.0 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
    t_p = 2.0 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)
    return t_s, t_p


# ==============================================================================
# Test 1: 圆偏振 → 椭圆偏振（Jones 矩阵）
# ==============================================================================

def test_circular_to_elliptical_polarization():
    """
    圆偏振 (S0=1, S1=0, S2=0, S3=1) 经 Fresnel 反射后
    变成椭圆偏振，S1 和 S3 由 r_s, r_p 决定：

        S1_out = (R_s − R_p) / (R_s + R_p)
        S3_out = 2 · r_s · r_p / (R_s + R_p)

    这是 Jones 矩阵 diag(r_s, r_p) 作用于圆偏振输入的直接后果。
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_i = math.radians(30.0)

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    # 覆盖为圆偏振 Stokes = (1, 0, 0, 1)
    rays["stokes"][0, 0] = 1.0
    rays["stokes"][0, 1] = 0.0
    rays["stokes"][0, 2] = 0.0
    rays["stokes"][0, 3] = 1.0

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    # 解析预期值
    r_s, r_p, _ = fresnel_rs_rp(n1, n2, theta_i)
    R_s, R_p = r_s ** 2, r_p ** 2
    S1_expected = (R_s - R_p) / (R_s + R_p)
    S3_expected = 2.0 * r_s * r_p / (R_s + R_p)

    # 反射发生在 step 1
    S1_out = out["stokes"][0, 1, 1].item()
    S3_out = out["stokes"][0, 1, 3].item()

    print(f"\n[Test 1] Circular → Elliptical")
    print(f"  θi = 30°, n1 = {n1}, n2 = {n2}")
    print(f"  r_s = {r_s:.6f}, r_p = {r_p:.6f}")
    print(f"  Expected: S1 = {S1_expected:.6f}, S3 = {S3_expected:.6f}")
    print(f"  Got     : S1 = {S1_out:.6f}, S3 = {S3_out:.6f}")

    assert abs(S1_out - S1_expected) < 1e-4, \
        f"S1 mismatch: got {S1_out}, expected {S1_expected}"
    assert abs(S3_out - S3_expected) < 1e-4, \
        f"S3 mismatch: got {S3_out}, expected {S3_expected}"

    # DoP 必须保持为 1（无退偏振）
    dop = math.sqrt(S1_out ** 2 + S3_out ** 2)
    assert abs(dop - 1.0) < 1e-4, f"DoP must remain 1, got {dop}"


# ==============================================================================
# Test 2: Fresnel 透射系数 t_s / t_p
# ==============================================================================

def test_fresnel_transmission_coefficients():
    """
    45° 线偏振输入经 Fresnel 折射后，输出偏振由 t_s、t_p 决定：

        S1_out = (t_s² − t_p²) / (t_s² + t_p²)
        S2_out = 2 · t_s · t_p / (t_s² + t_p²)

    这是 Jones 矩阵 diag(t_s, t_p) 作用于 45° 线偏振输入的直接后果。
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_i = math.radians(45.0)

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)   # 45° 线偏振 → S1=0, S2=1

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    # 解析预期值
    t_s, t_p = fresnel_ts_tp(n1, n2, theta_i)
    denom = t_s ** 2 + t_p ** 2
    S1_expected = (t_s ** 2 - t_p ** 2) / denom
    S2_expected = 2.0 * t_s * t_p / denom

    S1_out = out["stokes"][0, 1, 1].item()
    S2_out = out["stokes"][0, 1, 2].item()

    print(f"\n[Test 2] Fresnel Transmission Coefficients")
    print(f"  θi = 45°, n1 = {n1}, n2 = {n2}")
    print(f"  t_s = {t_s:.6f}, t_p = {t_p:.6f}")
    print(f"  Expected: S1 = {S1_expected:.6f}, S2 = {S2_expected:.6f}")
    print(f"  Got     : S1 = {S1_out:.6f}, S2 = {S2_out:.6f}")

    assert abs(S1_out - S1_expected) < 1e-4, \
        f"S1 mismatch: got {S1_out}, expected {S1_expected}"
    assert abs(S2_out - S2_expected) < 1e-4, \
        f"S2 mismatch: got {S2_out}, expected {S2_expected}"

    # 归一化检查
    assert abs(S1_out ** 2 + S2_out ** 2 - 1.0) < 1e-4, \
        f"S1² + S2² must = 1, got {S1_out ** 2 + S2_out ** 2}"


# ==============================================================================
# Test 3: 多次反射的偏振累积
# ==============================================================================

def test_polarization_multiple_bounce_flip():
    """
    45° 线偏振在正入射反射下，每次反射偏振翻转一次：

        Jones: (1/√2, 1/√2)  → 反射 → (r_s/√2, r_p/√2) = (r_s/√2, −r_s/√2)
                            → Stokes S2 = −1（变成 −45°）

    物理原因：正入射时 r_p = −r_s，符号差导致偏振轴旋转 90°。
    经过两次反射后回到原状态。

    这是 Stokes 参考系旋转的直接验证。
    """
    device = torch.device("cpu")
    verts, faces = make_two_parallel_mirrors(device, hs=5.0, z0=0.0, z1=1.0)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    # custom + reflectivity=1 → 每次命中都反射
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    S2_init = out["stokes"][0, 0, 2].item()   # 初始 45° → S2 = +1
    S2_1st = out["stokes"][0, 1, 2].item()    # 1 次反射后 → S2 翻转
    S2_2nd = out["stokes"][0, 2, 2].item()    # 2 次反射后 → S2 恢复
    S2_3rd = out["stokes"][0, 3, 2].item()    # 3 次反射后 → S2 再翻转

    print(f"\n[Test 3] Polarization Flip on Multiple Bounces")
    print(f"  Initial S2     = {S2_init:+.6f}  (+45°)")
    print(f"  After 1 bounce = {S2_1st:+.6f}  (flip → −45°)")
    print(f"  After 2 bounce = {S2_2nd:+.6f}  (recover → +45°)")
    print(f"  After 3 bounce = {S2_3rd:+.6f}  (flip → −45°)")

    assert abs(S2_init - 1.0) < 1e-5, \
        f"Initial S2 must be +1 (45°), got {S2_init}"
    assert abs(S2_1st + 1.0) < 1e-5, \
        f"After 1 bounce S2 must be −1 (flip), got {S2_1st}"
    assert abs(S2_2nd - 1.0) < 1e-5, \
        f"After 2 bounces S2 must be +1 (recover), got {S2_2nd}"
    assert abs(S2_3rd + 1.0) < 1e-5, \
        f"After 3 bounces S2 must be −1 (flip again), got {S2_3rd}"


# ==============================================================================
# Test 4: BK7 玻璃 d-line 参考折射率
# ==============================================================================

def test_bk7_d_line_reference_index():
    """
    N-BK7 玻璃在 d-line (587.6 nm) 的折射率参考值：n = 1.51680。

    参考：Schott Glass catalog (N-BK7 datasheet)

    验证 tracer 的波长插值返回与已发表参考值一致的结果。
    """
    device = torch.device("cpu")
    wl_d = 587.6
    n_ref = 1.51680
    n_sellmeier = bk7_sellmeier(wl_d)

    print(f"\n[Test 4] BK7 d-line Reference Index")
    print(f"  λ = {wl_d} nm")
    print(f"  Schott catalog reference: n = {n_ref:.5f}")
    print(f"  Sellmeier formula       : n = {n_sellmeier:.5f}")

    # 用 3 个波长点建立 LUT，其中包含 587.6nm 精确点
    n_spec = {
        "wavelengths_nm": [550.0, 587.6, 650.0],
        "values": [bk7_sellmeier(550.0), bk7_sellmeier(587.6), bk7_sellmeier(650.0)],
    }
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device,
        wavelength_nm=wl_d)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    n_measured = out["n2"][0, 1, 0].item()

    print(f"  Tracer output           : n = {n_measured:.5f}")

    # 与 Schott 参考值对比
    assert abs(n_measured - n_ref) < 1e-4, \
        f"tracer n(587.6nm) = {n_measured:.5f}, expected {n_ref:.5f} (Schott)"


# ==============================================================================
# Test 5: Sellmeier 色散曲线
# ==============================================================================

def test_sellmeier_dispersion_curve():
    """
    多个标准谱线下，tracer 的 n(λ) 与 BK7 Sellmeier 公式对比：

        F-line  (486.1 nm): n = 1.52238
        d-line  (587.6 nm): n = 1.51680
        C-line  (656.3 nm): n = 1.51432

    验证波长相关折射率的物理正确性。
    """
    device = torch.device("cpu")

    # 用密集网格构建 n(λ) LUT
    wl_grid = np.arange(400.0, 700.1, 20.0).tolist()
    n_values = [bk7_sellmeier(w) for w in wl_grid]
    n_spec = {
        "wavelengths_nm": wl_grid,
        "values": n_values,
    }
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print(f"\n[Test 5] Sellmeier Dispersion Curve")
    print(f"  {'λ (nm)':<12} {'n (Sellmeier)':<18} {'n (tracer)':<18} {'Δ':<12}")
    print(f"  " + "-" * 60)

    max_err = 0.0
    for wl, n_ref in sorted(BK7_REFERENCE.items()):
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device,
            wavelength_nm=wl)

        out = tracer.trace(rays, max_steps=3, surface_model="custom")
        n_meas = out["n2"][0, 1, 0].item()

        err = abs(n_meas - n_ref)
        max_err = max(max_err, err)
        print(f"  {wl:<12.1f} {n_ref:<18.5f} {n_meas:<18.5f} {err:<12.2e}")

    print(f"  " + "-" * 60)
    print(f"  Max error: {max_err:.2e}")

    assert max_err < 1e-4, \
        f"Sellmeier curve mismatch: max error = {max_err:.6e}"

    # ---- 色散趋势检查（n 应随 λ 增大而减小）----
    wls = [450.0, 550.0, 650.0]
    ns = []
    for wl in wls:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device,
            wavelength_nm=wl)
        out = tracer.trace(rays, max_steps=3, surface_model="custom")
        ns.append(out["n2"][0, 1, 0].item())

    print(f"\n  Dispersion trend: n(450)={ns[0]:.5f}, "
          f"n(550)={ns[1]:.5f}, n(650)={ns[2]:.5f}")
    assert ns[0] > ns[1] > ns[2], \
        f"Normal dispersion violated: n(450)={ns[0]}, n(550)={ns[1]}, n(650)={ns[2]}"


# ==============================================================================
# Test 6: Brewster 角偏振纯度
# ==============================================================================

def test_brewster_angle_polarization_purity():
    """
    Brewster 角 θ_B = atan(n₂/n₁) 处，R_p = 0。

    对于非偏振输入：
      - 反射光只能是 s 偏振
      - 反射光的 Stokes S1 应严格等于 +1

    这是 Brewster 角的经典性质：反射光完全线偏振（垂直于入射面）。
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_B = math.atan(n2 / n1)

    print(f"\n[Test 6] Brewster Angle Polarization Purity")
    print(f"  θ_B = atan({n2}/{n1}) = {math.degrees(theta_B):.4f}°")

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    direction = (math.sin(theta_B), 0.0, math.cos(theta_B))
    rays = generate_rays(
        source_type="grid", num_rays=2000,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="random", device=device)   # 非偏振输入

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    # 过滤出发生反射的光线
    # 反射的光线 transmissivity 低（大部分能量反射）
    T = out["transmissivity"][:, 1, 0]
    reflected_mask = T < 0.5
    n_reflected = reflected_mask.sum().item()

    print(f"  Total rays       : {rays['origins'].shape[0]}")
    print(f"  Reflected rays   : {n_reflected}")

    assert n_reflected > 20, \
        f"Too few reflected rays for statistics: {n_reflected}"

    # 反射光线的 S1 应该 ≈ 1（纯 s 偏振）
    S1_reflected = out["stokes"][reflected_mask, 1, 1]
    S1_min = S1_reflected.min().item()
    S1_mean = S1_reflected.mean().item()

    print(f"  S1 of reflected rays: min={S1_min:.6f}, mean={S1_mean:.6f}")

    # 允许小量数值偏差（Brewster 角附近 R_p 数值上 ≈ 1e-8）
    assert S1_min > 0.99, \
        f"Reflected light must be s-polarized (S1=1), got min S1={S1_min}"


# ==============================================================================
# Test 7: 色散 + 偏振 + 能量守恒的组合场景
# ==============================================================================

def test_dispersion_polarization_energy_combined():
    """
    组合场景：45° 线偏振光、45° 入射角、多个波长下穿过玻璃界面。

    验证：
      - 每个波长下 R + T + A = 1（能量守恒）
      - 透射偏振 S1 与解析 t_s(λ), t_p(λ) 一致
      - 不同波长的结果因色散而不同

    这个测试把三个独立物理效应（色散、偏振、能量守恒）耦合在一起。
    """
    device = torch.device("cpu")
    n1 = 1.0
    theta_i = math.radians(45.0)

    # 用 BK7 的 Sellmeier 建立 n(λ) LUT
    wl_grid = [450.0, 500.0, 550.0, 587.6, 600.0, 650.0, 700.0]
    n_values = [bk7_sellmeier(w) for w in wl_grid]
    n_spec = {"wavelengths_nm": wl_grid, "values": n_values}

    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))

    print(f"\n[Test 7] Dispersion + Polarization + Energy Combined")
    print(f"  θi = 45°, 45° linear polarization input")
    print(f"  {'λ (nm)':<10} {'n2':<12} {'S1 (expected)':<16} {'S1 (tracer)':<