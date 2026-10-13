"""
tests/test_deep_physics_validation.py
======================================
深度物理验证——7 个测试全部与解析解或已发表参考值对比。

Run:
    pytest tests/test_deep_physics_validation.py -v -s
"""
import math
import numpy as np
import pytest
import torch

from georay import AdvancedRayTracerPBRTGPU, generate_rays


# ==============================================================================
# 参考数据
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


BK7_REFERENCE = {
    486.1: 1.52238,
    587.6: 1.51680,
    656.3: 1.51432,
}


# ==============================================================================
# 几何辅助
# ==============================================================================

def make_flat_interface(device, hs=10.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def make_two_mirror_cavity(device, hs=5.0, z0=0.0, z1=1.0):
    """
    两个平行界面（z=z0, z=z1），用于测试光线在腔体内多次反射。

    ★ FM 布置（关键）：
        z=0 界面：fm[:, 0] = 0 (air, +z 侧即腔内侧)
                  fm[:, 1] = 1 (glass, -z 侧即外侧)
        z=1 界面：fm[:, 0] = 1 (glass, +z 侧即外侧)
                  fm[:, 1] = 0 (air, -z 侧即腔内侧)

    这样光线在腔体内（air）来回反射时，n1 始终是 air，
    n2 始终是 glass，Fresnel 系数 r_s, r_p 非零且每次符号一致。
    """
    verts = torch.tensor([
        [-hs, -hs, z0], [hs, -hs, z0], [hs, hs, z0], [-hs, hs, z0],
        [-hs, -hs, z1], [hs, -hs, z1], [hs, hs, z1], [-hs, hs, z1],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)

    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[0:2, 0] = 0
    fm[0:2, 1] = 1
    fm[2:4, 0] = 1
    fm[2:4, 1] = 0
    return verts, faces, fm


# ==============================================================================
# Fresnel 辅助
# ==============================================================================

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
# Test 1: 圆偏振 → 椭圆偏振（反射 Jones 矩阵）
# ==============================================================================

def test_circular_to_elliptical_polarization():
    """圆偏振 (1, 0, 0, 1) 经反射后变成椭圆偏振。"""
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

    rays["stokes"][0, 0] = 1.0
    rays["stokes"][0, 1] = 0.0
    rays["stokes"][0, 2] = 0.0
    rays["stokes"][0, 3] = 1.0

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    r_s, r_p, _ = fresnel_rs_rp(n1, n2, theta_i)
    R_s, R_p = r_s ** 2, r_p ** 2
    S1_expected = (R_s - R_p) / (R_s + R_p)
    S3_expected = 2.0 * r_s * r_p / (R_s + R_p)

    S1_out = out["stokes"][0, 1, 1].item()
    S3_out = out["stokes"][0, 1, 3].item()

    print("\n[Test 1] Circular -> Elliptical")
    print("  theta_i=30, n1={}, n2={}".format(n1, n2))
    print("  r_s={:.6f}, r_p={:.6f}".format(r_s, r_p))
    print("  Expected: S1={:.6f}, S3={:.6f}".format(S1_expected, S3_expected))
    print("  Got     : S1={:.6f}, S3={:.6f}".format(S1_out, S3_out))

    assert abs(S1_out - S1_expected) < 1e-4
    assert abs(S3_out - S3_expected) < 1e-4
    dop = math.sqrt(S1_out ** 2 + S3_out ** 2)
    assert abs(dop - 1.0) < 1e-4, "DoP must stay 1, got {}".format(dop)


# ==============================================================================
# Test 2: Fresnel 透射系数 t_s / t_p
# ==============================================================================

def test_fresnel_transmission_coefficients():
    """
    45° 线偏振输入经 Fresnel 折射，输出 Stokes 由 t_s、t_p 决定。

    ★ 修复：用 custom + reflectivity=0 → 100% 透射。
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_i = math.radians(45.0)

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    t_s, t_p = fresnel_ts_tp(n1, n2, theta_i)
    denom = t_s ** 2 + t_p ** 2
    S1_expected = (t_s ** 2 - t_p ** 2) / denom
    S2_expected = 2.0 * t_s * t_p / denom

    S1_out = out["stokes"][0, 1, 1].item()
    S2_out = out["stokes"][0, 1, 2].item()

    print("\n[Test 2] Fresnel Transmission Coefficients")
    print("  theta_i=45, n1={}, n2={}".format(n1, n2))
    print("  t_s={:.6f}, t_p={:.6f}".format(t_s, t_p))
    print("  Expected: S1={:.6f}, S2={:.6f}".format(S1_expected, S2_expected))
    print("  Got     : S1={:.6f}, S2={:.6f}".format(S1_out, S2_out))

    assert abs(S1_out - S1_expected) < 1e-4, \
        "S1 mismatch: got {}, expected {}".format(S1_out, S1_expected)
    assert abs(S2_out - S2_expected) < 1e-4, \
        "S2 mismatch: got {}, expected {}".format(S2_out, S2_expected)
    assert abs(S1_out ** 2 + S2_out ** 2 - 1.0) < 1e-4


# ==============================================================================
# Test 3: 多次反射的偏振翻转
# ==============================================================================

def test_polarization_multiple_bounce_flip():
    """
    45° 线偏振在正入射反射下每次翻转 S2 符号。

    ★ 修复：用 make_two_mirror_cavity 构造 air 腔体，
       使得每次反射 n1=air, n2=glass，r_s = -0.2, r_p = +0.2。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_two_mirror_cavity(device, hs=5.0, z0=0.0, z1=1.0)

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=4, surface_model="custom")

    S2_0 = out["stokes"][0, 0, 2].item()
    S2_1 = out["stokes"][0, 1, 2].item()
    S2_2 = out["stokes"][0, 2, 2].item()
    S2_3 = out["stokes"][0, 3, 2].item()

    print("\n[Test 3] Polarization Flip on Multiple Bounces")
    print("  Initial S2    = {:+.6f}  (+45 deg)".format(S2_0))
    print("  1st bounce S2 = {:+.6f}".format(S2_1))
    print("  2nd bounce S2 = {:+.6f}".format(S2_2))
    print("  3rd bounce S2 = {:+.6f}".format(S2_3))

    assert abs(S2_0 - 1.0) < 1e-5
    assert abs(S2_1 + 1.0) < 1e-5, "After 1 bounce S2 must be -1, got {}".format(S2_1)
    assert abs(S2_2 - 1.0) < 1e-5, "After 2 bounces S2 must be +1, got {}".format(S2_2)
    assert abs(S2_3 + 1.0) < 1e-5, "After 3 bounces S2 must be -1, got {}".format(S2_3)


# ==============================================================================
# Test 4: BK7 d-line 参考折射率
# ==============================================================================

def test_bk7_d_line_reference_index():
    """N-BK7 玻璃在 d-line (587.6 nm) 的折射率 = 1.51680。"""
    device = torch.device("cpu")
    wl_d = 587.6
    n_ref = 1.51680

    n_spec = {
        "wavelengths_nm": [550.0, 587.6, 650.0],
        "values": [bk7_sellmeier(550.0), bk7_sellmeier(587.6), bk7_sellmeier(650.0)],
    }
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
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
    print("\n[Test 4] BK7 d-line Reference Index")
    print("  Schott ref: n = {:.5f}".format(n_ref))
    print("  Tracer    : n = {:.5f}".format(n_measured))

    assert abs(n_measured - n_ref) < 1e-4


# ==============================================================================
# Test 5: Sellmeier 色散曲线
# ==============================================================================

def test_sellmeier_dispersion_curve():
    """多个标准谱线下 tracer 的 n(λ) 与 BK7 Sellmeier 对比。"""
    device = torch.device("cpu")

    wl_grid = np.arange(400.0, 700.1, 20.0).tolist()
    n_values = [bk7_sellmeier(w) for w in wl_grid]
    n_spec = {"wavelengths_nm": wl_grid, "values": n_values}
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 5] Sellmeier Dispersion Curve")
    print("  {:>10s} {:>18s} {:>18s} {:>12s}".format(
        "lambda", "n (Sellmeier)", "n (tracer)", "delta"))
    print("  " + "-" * 62)

    max_err = 0.0
    for wl, n_ref in sorted(BK7_REFERENCE.items()):
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=wl)
        out = tracer.trace(rays, max_steps=3, surface_model="custom")
        n_meas = out["n2"][0, 1, 0].item()
        err = abs(n_meas - n_ref)
        max_err = max(max_err, err)
        print("  {:>10.1f} {:>18.5f} {:>18.5f} {:>12.2e}".format(
            wl, n_ref, n_meas, err))

    print("  " + "-" * 62)
    print("  Max error: {:.2e}".format(max_err))
    assert max_err < 1e-4

    wls = [450.0, 550.0, 650.0]
    ns = []
    for wl in wls:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=wl)
        out = tracer.trace(rays, max_steps=3, surface_model="custom")
        ns.append(out["n2"][0, 1, 0].item())
    print("  Dispersion trend: n(450)={:.5f} > n(550)={:.5f} > n(650)={:.5f}".format(
        ns[0], ns[1], ns[2]))
    assert ns[0] > ns[1] > ns[2]


# ==============================================================================
# Test 6: Brewster 角偏振纯度
# ==============================================================================

def test_brewster_angle_polarization_purity():
    """
    Brewster 角 θ_B = atan(n₂/n₁) 处 R_p = 0，反射光完全 s 偏振。

    ★ 修复：按光线方向过滤反射光线，不用 transmissivity 字段。
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_B = math.atan(n2 / n1)

    print("\n[Test 6] Brewster Angle Polarization Purity")
    print("  theta_B = {:.4f} deg".format(math.degrees(theta_B)))

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

    # ★ 按方向过滤反射光线
    pos = out["positions"]
    active = out["active"]
    d_after = pos[:, 2] - pos[:, 1]
    norm = d_after.norm(dim=-1, keepdim=True)
    valid = (norm.squeeze(-1) > 1e-6) & active[:, 2]
    d_unit = torch.zeros_like(d_after)
    d_unit[valid] = d_after[valid] / norm[valid]

    reflected_mask = valid & (d_unit[:, 2] < 0)
    n_reflected = reflected_mask.sum().item()

    print("  Total rays     : {}".format(rays["origins"].shape[0]))
    print("  Reflected rays : {}".format(n_reflected))

    assert n_reflected > 20, \
        "Too few reflected rays: {}".format(n_reflected)

    S1_reflected = out["stokes"][reflected_mask, 1, 1]
    S1_min = S1_reflected.min().item()
    S1_mean = S1_reflected.mean().item()

    print("  S1 of reflected: min={:.6f}, mean={:.6f}".format(S1_min, S1_mean))
    assert S1_min > 0.99, \
        "Reflected light must be s-polarized (S1=1), got min S1={}".format(S1_min)


# ==============================================================================
# Test 7: 色散 + 偏振 + 能量守恒
# ==============================================================================

def test_dispersion_polarization_energy_combined():
    """
    色散 + 偏振 + 能量守恒的组合场景。

    ★ 修复：用 custom + reflectivity=0 → 100% 透射，
       这样比较的 Stokes 就是透射 Stokes，与解析解一致。
    """
    device = torch.device("cpu")
    n1 = 1.0
    theta_i = math.radians(45.0)

    wl_grid = [450.0, 500.0, 550.0, 587.6, 600.0, 650.0, 700.0]
    n_values = [bk7_sellmeier(w) for w in wl_grid]
    n_spec = {"wavelengths_nm": wl_grid, "values": n_values}

    materials = {
        0: {"n": n1, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))

    print("\n[Test 7] Dispersion + Polarization + Energy Combined")
    print("  theta_i = 45 deg, 45-degree linear input, 100% transmission")
    print("  {:>8s} {:>10s} {:>16s} {:>16s} {:>12s} {:>14s}".format(
        "lambda", "n2", "S1 (expected)", "S1 (tracer)", "|R+T+A-1|", "S2 (tracer)"))
    print("  " + "-" * 84)

    for wl in [486.1, 587.6, 656.3]:
        n2 = bk7_sellmeier(wl)

        t_s, t_p = fresnel_ts_tp(n1, n2, theta_i)
        denom = t_s ** 2 + t_p ** 2
        S1_expected = (t_s ** 2 - t_p ** 2) / denom

        rays = generate_rays(
            source_type="grid", num_rays=64,
            origin=(0.0, 0.0, -1.0), direction=direction,
            bounds=(-0.3, 0.3, -0.3, 0.3),
            polarization=45.0, device=device,
            wavelength_nm=wl)

        out = tracer.trace(rays, max_steps=3, surface_model="custom")

        S1_meas = out["stokes"][:, 1, 1].mean().item()
        S2_meas = out["stokes"][:, 1, 2].mean().item()

        R = out["reflectivity"][:, 1, 0]
        T = out["transmissivity"][:, 1, 0]
        A = out["absorptivity"][:, 1, 0]
        dev = (R + T + A - 1.0).abs().max().item()

        print("  {:>8.1f} {:>10.5f} {:>16.6f} {:>16.6f} {:>12.2e} {:>14.6f}".format(
            wl, n2, S1_expected, S1_meas, dev, S2_meas))

        assert abs(S1_meas - S1_expected) < 1e-3, \
            "lambda={}nm: S1 mismatch, got {:.6f}, expected {:.6f}".format(
                wl, S1_meas, S1_expected)
        assert dev < 1e-4, \
            "lambda={}nm: R+T+A-1 = {:.2e}".format(wl, dev)
        assert out["energy_conserved"], \
            "lambda={}nm: energy_conserved flag is False".format(wl)

    print("\n  -> S1 changes with wavelength due to dispersion")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))