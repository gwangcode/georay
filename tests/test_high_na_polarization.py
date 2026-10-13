"""
tests/test_high_na_polarization.py
===================================
高 NA 透镜的偏振像差验证。
"""
import math
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_sphere(device, radius=1.0, subdivisions=5):
    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    verts = torch.tensor(s.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(s.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def test_high_na_polarization_varies_with_angle():
    """
    高 NA 透镜边缘与近轴光线的偏振响应不同。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_sphere(device, radius=1.0, subdivisions=5)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 近轴光线
    rays_p = generate_rays(
        source_type="grid", num_rays=1,
        origin=(-2.0, 0.1, 0.0), direction=(1.0, 0.0, 0.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    # 大角度边缘光线
    rays_m = generate_rays(
        source_type="grid", num_rays=1,
        origin=(-2.0, 0.7, 0.0), direction=(1.0, 0.0, 0.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out_p = tracer.trace(rays_p, max_steps=3, surface_model="custom")
    out_m = tracer.trace(rays_m, max_steps=3, surface_model="custom")

    S1_p = out_p["stokes"][0, 1, 1].item()
    S1_m = out_m["stokes"][0, 1, 1].item()

    print("\n[Test 1] High NA polarization variation")
    print("  Paraxial S1: {:.6f}".format(S1_p))
    print("  Marginal S1: {:.6f}".format(S1_m))

    # 边缘光线偏振变化应更大
    assert abs(S1_m) > abs(S1_p), \
        "Marginal rays should show larger polarization change"


def test_high_na_dop_preserved():
    """高 NA 折射下 DoP 应保持为 1。"""
    device = torch.device("cpu")
    verts, faces, fm = make_sphere(device, radius=1.0, subdivisions=5)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=64,
        origin=(-2.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
        bounds=(-0.7, 0.7, -0.7, 0.7),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    dop = out["dop"][:, 1, 0]
    print("\n[Test 2] High NA DoP preserved")
    print("  DoP range: [{:.6f}, {:.6f}]".format(dop.min().item(), dop.max().item()))
    assert torch.allclose(dop, torch.ones_like(dop), atol=1e-3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))