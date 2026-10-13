"""
tests/test_performance_regression.py
=====================================
性能回归断言测试。

防止无声性能退化：
    * Warp 相对 PyTorch 的加速比低于阈值 → 失败
    * Warp 显存占用超过预算 → 失败
    * 单位吞吐量低于基线 → 失败

Run:
    pytest tests/test_performance_regression.py -v -s
"""
import time
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


_HAS_CUDA = torch.cuda.is_available()

try:
    import warp as wp
    _HAS_WARP = True
except ImportError:
    _HAS_WARP = False

try:
    from georay import WarpTracer as _WT
    _HAS_WARP_TRACER = _WT is not None
except ImportError:
    _HAS_WARP_TRACER = False


_skip_warp = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_WARP and _HAS_WARP_TRACER),
    reason="WarpTracer requires CUDA + warp-lang"
)


def make_scene(device, n_grid=4):
    """构建中等规模场景：盒 + n_grid² 个小球。"""
    meshes = [trimesh.creation.box(extents=[10.0, 10.0, 10.0])]
    for i in range(n_grid):
        for j in range(n_grid):
            s = trimesh.creation.icosphere(subdivisions=1, radius=0.3)
            s.apply_translation([-4.0 + 8.0 * i / max(n_grid - 1, 1),
                                 -4.0 + 8.0 * j / max(n_grid - 1, 1), 0.0])
            meshes.append(s)
    combined = trimesh.util.concatenate(meshes)
    verts = torch.tensor(combined.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    mats = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }
    return verts, faces, fm, mats, faces.shape[0]


def bench_pytorch(device, n_rays, n_grid=4, max_steps=20, warmup=2, repeat=3):
    verts, faces, fm, mats, _ = make_scene(device, n_grid)
    rays = generate_rays(
        source_type="grid", num_rays=n_rays,
        origin=(0.0, 0.0, -8.0), direction=(0.0, 0.0, 1.0),
        bounds=(-3.0, 3.0, -3.0, 3.0),
        polarization="s", device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, mats, device)
    for _ in range(warmup):
        tracer.trace(rays, max_steps=max_steps, surface_model="fresnel")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        tracer.trace(rays, max_steps=max_steps, surface_model="fresnel")
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000.0


def bench_warp(device, n_rays, n_grid=4, max_steps=20, warmup=2, repeat=5):
    verts, faces, fm, mats, _ = make_scene(device, n_grid)
    rays = generate_rays(
        source_type="grid", num_rays=n_rays,
        origin=(0.0, 0.0, -8.0), direction=(0.0, 0.0, 1.0),
        bounds=(-3.0,