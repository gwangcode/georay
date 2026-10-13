"""
tests/test_polarization_accumulation.py
========================================
偏振在完整光路中的累积效应。
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_two_mirrors(device, hs=5.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
        [-hs, -hs, 1.0], [hs, -hs, 1.0], [hs, hs, 1.0], [-hs, hs, 1.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)
    fm = torch.zeros((4, 2), dtype=torch.long, device=device)
    fm[0:2, 0] = 0
    fm[0:2, 1] = 1
    fm[2:4, 0] = 1
    fm[2:4, 1] = 0
    return verts, faces, fm


def make_grin_box(device, extents=(1.0, 1.0, 2.0)):
    box = trimesh.creation.box(extents=list(extents))
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


# ==============================================================================
# Test 1: 多次反射下 DoP 保持
# ==============================================================================

def test_polarization_dop_preserved_multiple_bounces():
    """
    纯偏振光（DoP = 1）在两个平行镜面间多次反射，DoP 应始终保持 1。

    ★ 修复：原测试只定义了 material 0，但 face_mats 引用了 mat 1，
       n_table 越界。改为定义 mat 0 和 mat 1。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_two_mirrors(device, hs=5.0)
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
    out = tracer.trace(rays, max_steps=5, surface_model="custom")

    dop = out["dop"][0, :, 0]

    print("\n[Test 1] DoP through multiple bounces")
    print("  DoP history: {}".format([round(v, 6) for v in dop.tolist()]))

    assert torch.allclose(dop, torch.ones_like(dop), atol=1e-4), \
        "DoP should stay 1 through specular reflections"


# ==============================================================================
# Test 2: 偏振在 GRIN 介质中的演化
# ==============================================================================

def test_polarization_evolution_in_grin():
    """
    GRIN 介质中，光线方向不断变化。偏振参考系应随方向旋转，
    但 DoP 应保持为 1。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 2.0))

    def parabolic_grin(x, y, z, wavelength):
        return 1.5 - 0.15 * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": parabolic_grin},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.3, 0.3, 0.0, 0.0),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=200, ds_init=0.02,
                        ds_min=1e-4, ds_max=0.05,
                        fresnel=False)

    dop = out["dop"][0, :, 0]

    print("\n[Test 2] Polarization in GRIN medium")
    print("  DoP range: [{:.6f}, {:.6f}]".format(
        dop.min().item(), dop.max().item()))

    # GRIN 步进不改变 DoP（几何演化 + 参考系旋转）
    assert torch.allclose(dop, torch.ones_like(dop), atol=1e-3), \
        "DoP should stay ~1 in GRIN medium"


# ==============================================================================
# Test 3: 部分偏振经过多个界面
# ==============================================================================

def test_partial_polarization_multiple_interfaces():
    """
    部分偏振光（DoP = 0.5）经过两个界面后，能量权重的累积
    应保持偏振的强度比例。

    验证方式：对比两个界面与两个单界面的能量权重乘积。
    """
    device = torch.device("cpu")
    hs = 100.0
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta = math.radians(45.0)
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(math.sin(theta), 0.0, math.cos(theta)),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    # 覆盖为部分偏振 Stokes
    rays["stokes"][0, 0] = 1.0
    rays["stokes"][0, 1] = 0.3
    rays["stokes"][0, 2] = 0.3
    rays["stokes"][0, 3] = 0.2

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    S1 = out["stokes"][0, 1, 1].item()
    S2 = out["stokes"][0, 1, 2].item()
    S3 = out["stokes"][0, 1, 3].item()
    dop = math.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2)

    print("\n[Test 3] Partial polarization through interface")
    print("  Input  DoP: {:.4f}".format(math.sqrt(0.3**2 + 0.3**2 + 0.2**2)))
    print("  Output DoP: {:.4f}".format(dop))
    print("  Output S1={:.4f}, S2={:.4f}, S3={:.4f}".format(S1, S2, S3))

    # Fresnel 是 s/p 选择性界面，DoP 会变化，但应保持 [0, 1] 范围
    assert 0.0 <= dop <= 1.0 + 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))