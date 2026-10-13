"""
tests/test_dispersive_grin.py
==============================
色散 GRIN 透镜的轴向色差验证。
"""
import numpy as np
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_grin_box(device, extents=(1.0, 1.0, 2.0)):
    box = trimesh.creation.box(extents=list(extents))
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def make_dispersive_grin_materials():
    """GRIN 介质：n0 随波长变化（色散），A 恒定。"""
    def dispersive_grin(x, y, z, wl):
        # wl: torch tensor [N, 1] in nm
        wl_scalar = wl if wl is not None else 550.0
        # n0(λ) = 1.55 - 0.0001 * (λ - 400)
        n0 = 1.55 - 0.0001 * (wl_scalar - 400.0)
        A = 0.15
        return n0 - A * (x ** 2 + y ** 2)

    return {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": dispersive_grin},
    }


def test_chromatic_focal_shift():
    """
    不同波长在同一 GRIN 透镜中聚焦位置不同（轴向色差）。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 2.0))
    materials = make_dispersive_grin_materials()

    x_finals = {}
    for wl in [400.0, 700.0]:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.3, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=wl)

        tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        out = tracer.trace(rays, max_steps=200, ds_init=0.01, fresnel=False)

        pos = out["positions"][0]
        x_finals[wl] = abs(pos[-1, 0].item())

    print("\n[Test 1] Chromatic focal shift")
    print("  |x_end(400nm)| = {:.4f}".format(x_finals[400.0]))
    print("  |x_end(700nm)| = {:.4f}".format(x_finals[700.0]))

    assert abs(x_finals[400.0] - x_finals[700.0]) > 1e-3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))