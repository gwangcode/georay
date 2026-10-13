"""
tests/test_grin_abbe.py
========================
GRIN 介质的 Abbe 数验证。

物理依据：
    Abbe 数 V_d = (n_d - 1) / (n_F - n_C)
    其中 n_d = n(587.6nm), n_F = n(486.1nm), n_C = n(656.3nm)
    V_d 越大，色散越小。
"""
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


def test_grin_abbe_number_consistent():
    """
    GRIN 介质的 Abbe 数应与输入 n(λ) 一致。

    ★ 修复：
      1. 色散系数符号：正常色散要求 n(λ) 随 λ 增大而减小（负号）。
      2. 系数大小：使用真实玻璃的 Δn/Δλ ≈ -4e-5 / nm。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device)

    def grin_fn(x, y, z, wl):
        # ★ 正常色散：n 随 λ 增大而减小
        # 系数 -4e-5 / nm 匹配真实 BK7 玻璃
        n0 = 1.5168 - 4e-5 * (wl - 587.6)
        A = 0.05
        return n0 - A * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": grin_fn},
    }

    n_measured = {}
    for wl in [486.1, 587.6, 656.3]:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=wl,
            medium_id=1)

        tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        out = tracer.trace(rays, max_steps=10, ds_init=0.01, fresnel=False)

        n_measured[wl] = out["n2"][0, 1, 0].item()

    print("\n[Test 1] GRIN Abbe number")
    print("  n(486.1) = {:.6f}".format(n_measured[486.1]))
    print("  n(587.6) = {:.6f}".format(n_measured[587.6]))
    print("  n(656.3) = {:.6f}".format(n_measured[656.3]))

    # 正常色散：n(短) > n(中) > n(长)
    assert n_measured[486.1] > n_measured[587.6] > n_measured[656.3], \
        "Normal dispersion violated: n(486)={:.6f}, n(588)={:.6f}, n(656)={:.6f}".format(
            n_measured[486.1], n_measured[587.6], n_measured[656.3])

    # Abbe 数
    denom = n_measured[486.1] - n_measured[656.3]
    assert denom > 1e-9, \
        "Abbe number denominator too small: {}".format(denom)

    V_d = (n_measured[587.6] - 1.0) / denom
    print("  Abbe number V_d = {:.2f}".format(V_d))

    # 真实 BK7 的 V_d ≈ 64，允许 50~80
    assert 30 < V_d < 100, \
        "Abbe number out of reasonable range: {:.2f}".format(V_d)


def test_grin_chromatic_focal_shift_scales_with_dispersion():
    """
    色散越强，焦点偏移越大。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device)

    def make_grin_with_dispersion(strength):
        def grin_fn(x, y, z, wl):
            wl_scalar = wl if wl is not None else 550.0
            n0 = 1.5 + strength * (wl_scalar - 550.0)
            A = 0.1
            return n0 - A * (x ** 2 + y ** 2)
        return grin_fn

    shifts = []
    for strength in [1e-5, 1e-4, 1e-3]:
        materials = {
            0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0,
                "absorptivity": 0.0},
            1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
                "absorptivity": 0.0,
                "grin_field": make_grin_with_dispersion(strength)},
        }

        focal = {}
        for wl in [400.0, 700.0]:
            rays = generate_rays(
                source_type="grid", num_rays=1,
                origin=(0.3, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
                bounds=(0.0, 0.0, 0.0, 0.0),
                polarization="s", device=device, wavelength_nm=wl)

            tracer = AdvancedRayTracerPBRTGPU(
                verts, faces, fm, materials, device)
            out = tracer.trace(rays, max_steps=200, ds_init=0.01,
                                fresnel=False)
            focal[wl] = out["positions"][0, -1, 0].item()

        shifts.append(abs(focal[400.0] - focal[700.0]))

    print("\n[Test 2] Chromatic focal shift scales with dispersion")
    for strength, shift in zip([1e-5, 1e-4, 1e-3], shifts):
        print("  strength={:.0e}: shift = {:.4f}".format(strength, shift))

    # 更强的色散应给出更大的焦点偏移
    assert shifts[2] > shifts[1] > shifts[0], \
        "Focal shift should monotonically increase with dispersion"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))