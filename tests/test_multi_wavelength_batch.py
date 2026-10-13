"""
tests/test_multi_wavelength_batch.py
=====================================
多波长批量追踪验证。
"""
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_flat_interface(device, hs=100.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def test_multi_wavelength_batch():
    """
    一次 trace 调用处理多波长光线（合并 ray_dict）。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)

    n_spec = {
        "wavelengths_nm": [400.0, 550.0, 700.0],
        "values": [1.55, 1.52, 1.49],
    }
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_spec, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 分别生成不同波长的光线
    rays_list = []
    for wl in [400.0, 550.0, 700.0]:
        rays = generate_rays(
            source_type="grid", num_rays=16,
            origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
            bounds=(-0.5, 0.5, -0.5, 0.5),
            polarization="s", device=device, wavelength_nm=wl)
        rays_list.append(rays)

    # 合并
    combined = {
        "origins": torch.cat([r["origins"] for r in rays_list], dim=0),
        "dirs": torch.cat([r["dirs"] for r in rays_list], dim=0),
        "wavelengths": torch.cat([r["wavelengths"] for r in rays_list], dim=0),
        "stokes": torch.cat([r["stokes"] for r in rays_list], dim=0),
        "s_ref": torch.cat([r["s_ref"] for r in rays_list], dim=0),
        "p_ref": torch.cat([r["p_ref"] for r in rays_list], dim=0),
        "ray_ids": torch.arange(48, dtype=torch.long, device=device),
        "medium_ids": torch.zeros(48, dtype=torch.long, device=device),
    }

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(combined, max_steps=3, surface_model="custom")

    # 三组波长应给出不同的 n2
    n2_400 = out["n2"][0, 1, 0].item()
    n2_700 = out["n2"][32, 1, 0].item()

    print("\n[Test 1] Multi-wavelength batch")
    print("  n2(400nm) = {:.4f}".format(n2_400))
    print("  n2(700nm) = {:.4f}".format(n2_700))

    assert abs(n2_400 - 1.55) < 1e-3
    assert abs(n2_700 - 1.49) < 1e-3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))