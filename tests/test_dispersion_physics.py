"""
tests/test_dispersion_physics.py
=================================
色散的物理真实性：Sellmeier 方程、Cauchy 公式、色散焦移。
"""
import math
import numpy as np
import pytest
import torch

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def bk7_sellmeier(wl_nm):
    """N-BK7 玻璃的 Sellmeier 方程。"""
    wl_um2 = (wl_nm * 1e-3) ** 2
    B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
    C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653
    return math.sqrt(1.0 + B1*wl_um2/(wl_um2 - C1)
                     + B2*wl_um2/(wl_um2 - C2)
                     + B3*wl_um2/(wl_um2 - C3))


def cauchy_n(wl_nm, A=1.5, B=5000.0):
    """Cauchy 公式 n(λ) = A + B/λ²。"""
    return A + B / (wl_nm ** 2)


def make_flat_interface(device, hs=100.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


# ==============================================================================
# Test 1: Sellmeier 方程匹配
# ==============================================================================

def test_sellmeier_matches_analytic():
    """
    用密集的 BK7 Sellmeier LUT，tracer 输出的 n(λ) 应与解析解一致。
    """
    device = torch.device("cpu")
    wl_grid = np.arange(400.0, 700.1, 10.0).tolist()
    n_values = [bk7_sellmeier(w) for w in wl_grid]

    n_spec = {"wavelengths_nm": wl_grid, "values": n_values}
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    max_err = 0.0
    test_wls = [450.0, 500.0, 550.0, 587.6, 600.0, 650.0]

    print("\n[Test 1] Sellmeier equation match")
    print("  {:>8s} {:>14s} {:>14s} {:>10s}".format(
        "lambda", "n_expected", "n_tracer", "error"))
    print("  " + "-" * 50)

    for wl in test_wls:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=wl)
        out = tracer.trace(rays, max_steps=3, surface_model="custom")
        n_meas = out["n2"][0, 1, 0].item()
        n_exact = bk7_sellmeier(wl)
        err = abs(n_meas - n_exact)
        max_err = max(max_err, err)
        print("  {:>8.1f} {:>14.6f} {:>14.6f} {:>10.2e}".format(
            wl, n_exact, n_meas, err))

    assert max_err < 1e-4


# ==============================================================================
# Test 2: Cauchy 公式匹配
# ==============================================================================

def test_cauchy_matches_analytic():
    """
    Cauchy 公式 n(λ) = A + B/λ² 也应被 tracer 正确处理。
    """
    device = torch.device("cpu")
    A, B = 1.5, 5000.0
    wl_grid = np.arange(400.0, 700.1, 10.0).tolist()
    n_values = [cauchy_n(w, A, B) for w in wl_grid]

    n_spec = {"wavelengths_nm": wl_grid, "values": n_values}
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    max_err = 0.0
    print("\n[Test 2] Cauchy equation match")
    for wl in [400.0, 500.0, 600.0, 700.0]:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=wl)
        out = tracer.trace(rays, max_steps=3, surface_model="custom")
        n_meas = out["n2"][0, 1, 0].item()
        n_exact = cauchy_n(wl, A, B)
        err = abs(n_meas - n_exact)
        max_err = max(max_err, err)
        print("  λ={:.0f}nm: n_exact={:.6f}, n_meas={:.6f}, err={:.2e}".format(
            wl, n_exact, n_meas, err))

    assert max_err < 1e-4


# ==============================================================================
# Test 3: 色散导致的 Snell 角变化
# ==============================================================================

def test_chromatic_snell_angle():
    """
    不同波长的光在界面处应有不同的折射角（因为 n(λ) 不同）。
    """
    device = torch.device("cpu")
    n_spec = {
        "wavelengths_nm": [400.0, 550.0, 700.0],
        "values": [1.55, 1.52, 1.49],
    }
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    theta_i = math.radians(30.0)
    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))

    refracted_angles = {}
    for wl in [400.0, 550.0, 700.0]:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=direction,
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=wl)
        out = tracer.trace(rays, max_steps=3, surface_model="custom")
        pos = out["positions"][0]
        d_after = (pos[2] - pos[1])
        d_after = d_after / d_after.norm()
        cos_t = abs(d_after[2].item())
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t ** 2))
        refracted_angles[wl] = math.degrees(math.asin(sin_t))

    print("\n[Test 3] Chromatic Snell angle")
    for wl, ang in refracted_angles.items():
        print("  λ={:.0f}nm: θt = {:.4f}°".format(wl, ang))

    # 短波长 (n 大) 折射角应小于长波长
    assert refracted_angles[400.0] < refracted_angles[700.0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))