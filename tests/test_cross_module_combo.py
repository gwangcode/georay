"""
tests/test_cross_module_combo.py
=================================
跨模块组合场景：GRIN + 偏振 + 色散 + 能量守恒。
"""
import math
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_grin_box(device, extents=(1.0, 1.0, 4.0)):
    box = trimesh.creation.box(extents=list(extents))
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def test_grin_polarization_dispersion_combined():
    """
    同时启用 GRIN + 偏振 + 色散。
    验证：能量守恒、DoP 保持、色散导致不同焦点。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device)

    def dispersive_grin(x, y, z, wl):
        # n0(λ) 随波长变化，A 恒定
        wl_scalar = wl if wl is not None else 550.0
        n0 = 1.55 - 0.0001 * (wl_scalar - 400.0)
        A = 0.1
        return n0 - A * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": dispersive_grin},
    }

    focal_positions = {}
    for wl in [400.0, 700.0]:
        rays = generate_rays(
            source_type="grid", num_rays=8,
            origin=(0.3, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization=45.0, device=device, wavelength_nm=wl)

        tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        out = tracer.trace(rays, max_steps=200, ds_init=0.02,
                            ds_min=1e-4, ds_max=0.05, fresnel=False)

        x_end = out["positions"][:, -1, 0].mean().item()
        dop = out["dop"][:, -1, 0].mean().item()

        focal_positions[wl] = abs(x_end)

        print("\n[Test 1] GRIN + Pol + Dispersion (λ={}nm)".format(wl))
        print("  Mean |x_end|: {:.4f}".format(abs(x_end)))
        print("  Mean DoP    : {:.4f}".format(dop))
        print("  Energy conserved: {}".format(out["energy_conserved"]))

        # 能量守恒
        assert out["energy_conserved"]
        # DoP 保持
        assert dop > 0.99

    # 色散导致不同波长焦点不同
    diff = abs(focal_positions[400.0] - focal_positions[700.0])
    print("\n  |x_end(400) - x_end(700)| = {:.4f}".format(diff))
    assert diff > 1e-3, "Dispersion should cause different focal positions"


def test_full_pipeline_energy_conservation():
    """
    复杂场景下 R + T + A = 1 在每个界面成立。
    场景：立方体 + 界面 + 偏振 + 吸收。
    """
    device = torch.device("cpu")
    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.05},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.05},
    }

    rays = generate_rays(
        source_type="grid", num_rays=64,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

    R = out["reflectivity"]
    T = out["transmissivity"]
    A = out["absorptivity"]
    total = R + T + A
    interacted = total > 1e-6

    if interacted.any():
        max_dev = (total[interacted] - 1.0).abs().max().item()
        print("\n[Test 2] Full pipeline energy conservation")
        print("  Interfaces: {}".format(interacted.sum().item()))
        print("  max |R+T+A-1|: {:.3e}".format(max_dev))
        assert max_dev < 1e-4


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))