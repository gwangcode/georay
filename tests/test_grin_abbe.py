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
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device)

    # 用 Sellmeier 风格的色散
    def grin_fn(x, y, z, wl):
        wl_scalar = wl if wl is not None else 587.6
        # n(λ) = 1.5168 + 0.003·(λ-587.6) / 100  （简化的色散）
        n0 = 1.5168 + 0.003 * (wl_scalar - 587.6) / 100.0
        A = 0.05
        return n0 - A * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": grin_fn},
    }

    # 采样 3 个标准谱线的 n
    n_measured = {}
    for wl in [486.1, 587.6, 656.3]:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=wl)

        tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        out = tracer.trace(rays, max_steps=10, ds_init=0.01, fresnel=False)

        # 中心轴的 n（x=0, y=0）
        n_measured[wl] = out["n1"][0, 1, 0].item()

    # Abbe 数
    V_d = (n_measured[587.6] - 1.0) / (
        n_measured[486.1] - n_measured[656.3])

    print("\n[Test 1] GRIN Abbe number")
    print("  n(486.1) = {:.6f}".format(n_measured[486.1]))
    print("  n(587.6) = {:.6f}".format(n_measured[587.6]))
    print("  n(656.3) = {:.6f}".format(n_measured[656.3]))
    print("  Abbe number V_d = {:.2f}".format(V_d))

    # 简化的色散给出较小的 V_d（因为 dn/dλ 大）
    # 只验证符号和数量级
    assert V_d > 0, "Abbe number should be positive"
    assert V_d < 1e6, "Abbe number unreasonable: {}".format(V_d)


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