"""
tests/test_large_scale_bvh.py
==============================
大规模网格下 BVH 的性能与精度。
"""
import time
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_large_mesh(device, n_spheres=100):
    """构造包含 n_spheres 个球的大规模场景。"""
    meshes = [trimesh.creation.box(extents=[20.0, 20.0, 20.0])]
    import math
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
    """大规模网格下，光线仍应精确命中。"""
    device = torch.device("cpu")
    verts, faces, fm, _ = make_large_mesh(device, n_spheres=50)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 中心光线应命中盒子正面
    rays = generate_rays(
        source_type="grid", num_rays=64,
        origin=(0.0, 0.0, -15.0), direction=(0.0, 0.0, 1.0),
        bounds=(-3.0, 3.0, -3.0, 3.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="custom")

    # 所有光线都应命中（在 z=0 附近的球或 z=-10 的盒子面）
    hit_fraction = out["active"][:, 0].float().mean().item()

    print("\n[Test 2] Large mesh intersect")
    print("  Hit fraction at step 0: {:.2f}".format(hit_fraction))

    assert hit_fraction > 0.9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))