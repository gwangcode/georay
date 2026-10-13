"""
tests/test_mesh_robustness.py
==============================
网格鲁棒性：零面积三角形、非流形、内部起点、极端长宽比。
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


# ==============================================================================
# Test 1: 零面积三角形
# ==============================================================================

def test_zero_area_triangle_handling():
    """网格中包含退化三角形时，tracer 不应崩溃。"""
    device = torch.device("cpu")

    # 有效三角形 + 退化三角形（三点共线）
    verts = torch.tensor([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0],  # 共线，退化
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],   # 有效
        [4, 5, 6],               # 退化
    ], dtype=torch.long, device=device)
    fm = torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.long, device=device)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    try:
        tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        rays = generate_rays(
            source_type="grid", num_rays=4,
            origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
            bounds=(-0.5, 0.5, -0.5, 0.5),
            polarization="s", device=device)
        out = tracer.trace(rays, max_steps=5)

        assert not torch.isnan(out["positions"]).any()
        print("\n[Test 1] Zero-area triangle: gracefully handled")
    except Exception as e:
        pytest.fail("Zero-area triangle caused crash: {}".format(e))


# ==============================================================================
# Test 2: 非流形网格
# ==============================================================================

def test_non_manifold_mesh():
    """边缘被多于两个面共享时，BVH 应正确遍历。"""
    device = torch.device("cpu")

    # 三个三角形共享一条边 (0, 1)
    verts = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],   # 共享边
        [0.5, 1.0, 0.0],                     # 三角形 A
        [0.5, -1.0, 0.0],                    # 三角形 B
        [0.5, 0.0, 1.0],                     # 三角形 C
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 1, 3], [0, 1, 4],
    ], dtype=torch.long, device=device)
    fm = torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.long, device=device)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    rays = generate_rays(
        source_type="grid", num_rays=4,
        origin=(0.5, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.2, 0.2, -0.2, 0.2),
        polarization="s", device=device)
    out = tracer.trace(rays, max_steps=5)

    assert not torch.isnan(out["positions"]).any()
    print("\n[Test 2] Non-manifold mesh handled")


# ==============================================================================
# Test 3: 光线从网格内部出发
# ==============================================================================

def test_ray_starts_inside_mesh():
    """光线从立方体内部出发，应命中内表面。"""
    device = torch.device("cpu")
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 从立方体内部 (0, 0, 0) 出发
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device,
        medium_id=1)   # ★ 声明从玻璃内部出发

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5)

    # 第一次 step 应命中 z=1 面
    hit_pos = out["positions"][0, 1]
    print("\n[Test 3] Ray from inside mesh")
    print("  Hit position: {}".format([round(v, 4) for v in hit_pos.tolist()]))

    assert abs(hit_pos[2].item() - 1.0) < 1e-3, \
        "Ray should hit z=1 face, got z = {:.4f}".format(hit_pos[2].item())


# ==============================================================================
# Test 4: 极端长宽比三角形
# ==============================================================================

def test_extreme_aspect_ratio_triangles():
    """三角形极度细长时，求交精度应保持。"""
    device = torch.device("cpu")

    # 三角形：宽 1000，高 0.001
    verts = torch.tensor([
        [-500.0, -0.0005, 0.0],
        [500.0, -0.0005, 0.0],
        [0.0, 0.0005, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    fm = torch.tensor([[1, 0]], dtype=torch.long, device=device)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    out = tracer.trace(rays, max_steps=3)

    hit_dist = out["positions"][0, 1, 2].item()

    print("\n[Test 4] Extreme aspect ratio triangle")
    print("  Hit z-coordinate: {:.6f} (expected 0.0)".format(hit_dist))

    assert abs(hit_dist) < 1e-4, \
        "Slender triangle: hit z = {:.6f}, expected 0".format(hit_dist)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))