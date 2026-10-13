"""
tests/test_large_scale_bvh.py
==============================
大规模网格下 BVH 的性能与精度。
"""
import time
import math
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_large_mesh(device, n_spheres=100):
    """构造包含 n_spheres 个球的大规模场景。"""
    meshes = [trimesh.creation.box(extents=[20.0, 20.0, 20.0])]
    for k in range(n_spheres):
        s = trimesh.creation.icosphere(subdivisions=1, radius=0.3)
        theta = 2 * math.pi * k / n_spheres
        r = 5.0
        s.apply_translation([r * math.cos(theta), r * math.sin(theta), 0.0])
        meshes.append(s)
    combined = trimesh.util.concatenate(meshes)
    verts = torch.tensor(combined.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm, faces.shape[0]


def test_large_mesh_build_time():
    """100 球场景的 BVH 构建应在 10 秒内完成。"""
    device = torch.device("cpu")
    verts, faces, fm, n_faces = make_large_mesh(device, n_spheres=100)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    print("\n[Test 1] Large mesh build")
    print("  Faces: {}".format(n_faces))

    t0 = time.perf_counter()
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    build_time = time.perf_counter() - t0

    print("  Build time: {:.3f} s".format(build_time))

    assert build_time < 10.0, \
        "BVH build took {:.2f}s > 10s".format(build_time)


def test_large_mesh_intersect_correctness():
    """
    大规模网格下，光线应精确命中。

    ★ 修复：
      1. 原测试检查 active[:, 0]（永远为 True），逻辑无意义。
         改为检查光线在 step 1 后有真实位移（position 变化）。
      2. 原 bounds 覆盖不到环形分布的球，改为 ±6 覆盖半径 5 的整个环。
    """
    device = torch.device("cpu")
    verts, faces, fm, _ = make_large_mesh(device, n_spheres=50)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 从 -z 射入，覆盖整个环形球分布
    rays = generate_rays(
        source_type="grid", num_rays=256,
        origin=(0.0, 0.0, -15.0), direction=(0.0, 0.0, 1.0),
        bounds=(-6.0, 6.0, -6.0, 6.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="custom")

    # ★ 检查 pos[1] != pos[0]（光线确实移动了，说明有求交）
    pos0 = out["positions"][:, 0, :]
    pos1 = out["positions"][:, 1, :]
    moved = (pos1 - pos0).norm(dim=-1) > 1e-3
    hit_fraction = moved.float().mean().item()

    print("\n[Test 2] Large mesh intersect")
    print("  Rays moved after step 0: {:.4f}".format(hit_fraction))

    # 大部分光线应命中（除了从盒子外穿出的少数）
    assert hit_fraction > 0.9, \
        "Too few rays moved: {:.4f}".format(hit_fraction)

    # 额外检查：确认有反射发生（R 或 T 非零）
    R = out["reflectivity"][:, 1, 0]
    T = out["transmissivity"][:, 1, 0]
    events = (R + T) > 1e-6
    n_events = events.sum().item()

    print("  Interface events: {} / {}".format(n_events, rays["origins"].shape[0]))
    assert n_events > 0, "No interface events detected"