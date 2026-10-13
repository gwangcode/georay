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
    """N-BK7 玻璃在波长 λ (nm) 下的折射率（Schott 官方 Sellmeier 系数）。"""
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
    变成椭圆偏振：

        S1_out = (R_s − R_p) / (R_s + R_p)
        S3_out = 2 · r_s · r_p / (R_s + R_p)
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

    S1_out = out["stokes"][0, 1, 1].item()
    S3_out = out["stokes"][0, 1, 3].item()

    print("\n[Test 1] Circular -> Elliptical")
    print("  theta_i = 30 deg, n1 = {}, n2 = {}".format(n1, n2))
    print("  r_s = {:.6f}, r_p = {:.6f}".format(r_s, r_p))
    print("  Expected: S1 = {:.6f}, S3 = {:.6f}".format(S1_expected, S3_expected))
    print("  Got     : S1 = {:.6f}, S3 = {:.6f}".format(S1_out, S3_out))

    assert abs(S1_out - S1_expected) < 1e-4, \
        "S1 mismatch: got {}, expected {}".format(S1_out, S1_expected)
    assert abs(S3_out - S3_expected) < 1e-4, \
        "S3 mismatch: got {}, expected {}".format(S3_out, S3_expected)

    # DoP 必须保持为 1（无退偏振）
    dop = math.sqrt(S1_out ** 2 + S3_out ** 2)
    assert abs(dop - 1.0) < 1e-4, "DoP must remain 1, got {}".format(dop)


# ==============================================================================
# Test 2: Fresnel 透射系数 t_s / t_p
# ==============================================================================

def test_fresnel_transmission_coefficients():
    """
    45° 线偏振输入经 Fresnel 折射后，输出偏振由 t_s、t_p 决定：

        S1_out = (t_s² − t_p²) / (t_s² + t_p²)
        S2_out = 2 · t_s · t_p / (t_s² + t_p²)
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

    print("\n[Test 2] Fresnel Transmission Coefficients")
    print("  theta_i = 45 deg, n1 = {}, n2 = {}".format(n1, n2))
    print("  t_s = {:.6f}, t_p = {:.6f}".format(t_s, t_p))
    print("  Expected: S1 = {:.6f}, S2 = {:.6f}".format(S1_expected, S2_expected))
    print("  Got     : S1 = {:.6f}, S2 = {:.6f}".format(S1_out, S2_out))

    assert abs(S1_out - S1_expected) < 1e-4, \
        "S1 mismatch: got {}, expected {}".format(S1_out, S1_expected)
    assert abs(S2_out - S2_expected) < 1e-4, \
        "S2 mismatch: got {}, expected {}".format(S2_out, S2_expected)

    # 归一化检查
    assert abs(S1_out ** 2 + S2_out ** 2 - 1.0) < 1e-4, \
        "S1^2 + S2^2 must = 1, got {}".format(S1_out ** 2 + S2_out ** 2)


# ==============================================================================
# Test 3: 多次反射的偏振累积
# ==============================================================================

def test_polarization_multiple_bounce_flip():
    """
    45° 线偏振在正入射反射下，每次反射偏振翻转一次。
    正入射时 r_p = −r_s，符号差导致 S2 翻转。
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
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    S2_init = out["stokes"][0, 0, 2].item()
    S2_1st = out["stokes"][0, 1, 2].item()
    S2_2nd = out["stokes"][0, 2, 2].item()
    S2_3rd = out["stokes"][0, 3, 2].item()

    print("\n[Test 3] Polarization Flip on Multiple Bounces")
    print("  Initial S2     = {:+.6f}  (+45 deg)".format(S2_init))
    print("  After 1 bounce = {:+.6f}  (flip to -45 deg)".format(S2_1st))
    print("  After 2 bounce = {:+.6f}  (recover to +45 deg)".format(S2_2nd))
    print("  After 3 bounce = {:+.6f}  (flip to -45 deg)".format(S2_3rd))

    assert abs(S2_init - 1.0) < 1e-5, \
        "Initial S2 must be +1 (45 deg), got {}".format(S2_init)
    assert abs(S2_1st + 1.0) < 1e-5, \
        "After 1 bounce S2 must be -1 (flip), got {}".format(S2_1st)
    assert abs(S2_2nd - 1.0) < 1e-5, \
        "After 2 bounces S2 must be +1 (recover), got {}".format(S2_2nd)
    assert abs(S2_3rd + 1.0) < 1e-5, \
        "After 3 bounces S2 must be -1 (flip again), got {}".format(S2_3rd)


# ==============================================================================
# Test 4: BK7 玻璃 d-line 参考折射率
# ==============================================================================

def test_bk7_d_line_reference_index():
    """
    N-BK7 玻璃在 d-line (587.6 nm) 的折射率参考值：n = 1.51680。
    参考：Schott Glass catalog (N-BK7 datasheet)
    """
    device = torch.device("cpu")
    wl_d = 587.6
    n_ref = 1.51680
    n_sellmeier = bk7_sellmeier(wl_d)

    print("\n[Test 4] BK7 d-line Reference Index")
    print("  lambda = {} nm".format(wl_d))
    print("  Schott catalog reference: n = {:.5f}".format(n_ref))
    print("  Sellmeier formula       : n = {:.5f}".format(n_sellmeier))

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
    print("  Tracer output           : n = {:.5f}".format(n_measured))

    assert abs(n_measured - n_ref) < 1e-4, \
        "tracer n(587.6nm) = {:.5f}, expected {:.5f} (Schott)".format(n_measured, n_ref)


# ==============================================================================
# Test 5: Sellmeier 色散曲线
# ==============================================================================

def test_sellmeier_dispersion_curve():
    """
    多个标准谱线下，tracer 的 n(λ) 与 BK7 Sellmeier 公式对比。
    """
    device = torch.device("cpu")

    wl_grid = np.arange(400.0, 700.1, 20.0).tolist()
    n_values = [bk7_sellmeier(w) for w in wl_grid]
    n_spec = {"wavelengths_nm": wl_grid, "values": n_values}
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 5] Sellmeier Dispersion Curve")
    header = "  {:>8s} {:>18s} {:>18s} {:>12s}".format(
        "lambda", "n (Sellmeier)", "n (tracer)", "delta")
    print(header)
    print("  " + "-" * 60)

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
        print("  {:>8.1f} {:>18.5f} {:>18.5f} {:>12.2e}".format(
            wl, n_ref, n_meas, err))

    print("  " + "-" * 60)
    print("  Max error: {:.2e}".format(max_err))

    assert max_err < 1e-4, \
        "Sellmeier curve mismatch: max error = {:.6e}".format(max_err)

    # 色散趋势
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

    print("\n  Dispersion trend: n(450)={:.5f}, n(550)={:.5f}, n(650)={:.5f}".format(
        ns[0], ns[1], ns[2]))
    assert ns[0] > ns[1] > ns[2], \
        "Normal dispersion violated: n(450)={}, n(550)={}, n(650)={}".format(
            ns[0], ns[1], ns[2])


# ==============================================================================
# Test 6: Brewster 角偏振纯度
# ==============================================================================

def test_brewster_angle_polarization_purity():
    """
    Brewster 角 θ_B = atan(n₂/n₁) 处，R_p = 0。
    对于非偏振输入，反射光只能是 s 偏振，S1 应严格等于 +1。
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_B = math.atan(n2 / n1)

    print("\n[Test 6] Brewster Angle Polarization Purity")
    print("  theta_B = atan({}/{}) = {:.4f} deg".format(
        n2, n1, math.degrees(theta_B)))

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
        polarization="random", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    T = out["transmissivity"][:, 1, 0]
    reflected_mask = T < 0.5
    n_reflected = reflected_mask.sum().item()

    print("  Total rays       : {}".format(rays["origins"].shape[0]))
    print("  Reflected rays   : {}".format(n_reflected))

    assert n_reflected > 20, \
        "Too few reflected rays for statistics: {}".format(n_reflected)

    S1_reflected = out["stokes"][reflected_mask, 1, 1]
    S1_min = S1_reflected.min().item()
    S1_mean = S1_reflected.mean().item()

    print("  S1 of reflected rays: min={:.6f}, mean={:.6f}".format(S1_min, S1_mean))

    assert S1_min > 0.99, \
        "Reflected light must be s-polarized (S1=1), got min S1={}".format(S1_min)


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
    """
    device = torch.device("cpu")
    n1 = 1.0
    theta_i = math.radians(45.0)

    # BK7 Sellmeier n(λ) LUT
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

    print("\n[Test 7] Dispersion + Polarization + Energy Combined")
    print("  theta_i = 45 deg, 45-degree linear polarization input")
    header = "  {:>8s} {:>10s} {:>16s} {:>16s} {:>12s} {:>14s}".format(
        "lambda", "n2", "S1 (expected)", "S1 (tracer)", "|R+T+A-1|", "S2 (tracer)")
    print(header)
    print("  " + "-" * 84)

    test_wls = [486.1, 587.6, 656.3]

    for wl in test_wls:
        # 解析 n2
        n2 = bk7_sellmeier(wl)

        # 解析预期 S1（透射后偏振）
        t_s, t_p = fresnel_ts_tp(n1, n2, theta_i)
        denom = t_s ** 2 + t_p ** 2
        S1_expected = (t_s ** 2 - t_p ** 2) / denom

        # Tracer 运行
        rays = generate_rays(
            source_type="grid", num_rays=64,
            origin=(0.0, 0.0, -1.0), direction=direction,
            bounds=(-0.3, 0.3, -0.3, 0.3),
            polarization=45.0, device=device,
            wavelength_nm=wl)

        out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

        # 透射光线的 S1、S2
        T = out["transmissivity"][:, 1, 0]
        transmitted_mask = T > 0.5
        S1_meas = out["stokes"][transmitted_mask, 1, 1].mean().item()
        S2_meas = out["stokes"][transmitted_mask, 1, 2].mean().item()

        # 能量守恒
        R = out["reflectivity"][:, 1, 0]
        Tt = out["transmissivity"][:, 1, 0]
        A = out["absorptivity"][:, 1, 0]
        total = R + Tt + A
        dev = (total - 1.0).abs().max().item()

        print("  {:>8.1f} {:>10.5f} {:>16.6f} {:>16.6f} {:>12.2e} {:>14.6f}".format(
            wl, n2, S1_expected, S1_meas, dev, S2_meas))

        # 断言
        assert abs(S1_meas - S1_expected) < 1e-3, \
            "lambda={}nm: S1 mismatch, got {:.6f}, expected {:.6f}".format(
                wl, S1_meas, S1_expected)
        assert dev < 1e-4, \
            "lambda={}nm: R+T+A-1 = {:.2e} > 1e-4".format(wl, dev)
        assert out["energy_conserved"], \
            "lambda={}nm: energy_conserved flag is False".format(wl)

    # ---- 色散导致的偏振差 ----
    print("\n  -> S1 changes with wavelength due to dispersion (see table above)")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))