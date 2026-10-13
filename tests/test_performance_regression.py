"""
tests/test_performance_regression.py
=====================================
性能回归监控测试。
"""
import time
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays

try:
    from georay import WarpTracer
    _HAS_WARP_TRACER = WarpTracer is not None
except ImportError:
    WarpTracer = None
    _HAS_WARP_TRACER = False

_HAS_CUDA = torch.cuda.is_available()

try:
    import warp as wp
    _HAS_WARP = True
except ImportError:
    _HAS_WARP = False


_skip_warp = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_WARP and _HAS_WARP_TRACER),
    reason="WarpTracer requires CUDA + warp-lang + georay[warp]"
)


def make_scene(device, n_grid=4):
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
    return verts, faces, fm, mats


def make_rays(device, N):
    return generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -8.0), direction=(0.0, 0.0, 1.0),
        bounds=(-3.0, 3.0, -3.0, 3.0),
        polarization="s", device=device)


@_skip_warp
def test_warp_speedup_regression():
    device = torch.device("cuda")
    verts, faces, fm, mats = make_scene(device, n_grid=4)
    rays = make_rays(device, N=1024)

    tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, mats, device)
    tp.trace(rays, max_steps=2, surface_model="fresnel")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tp.trace(rays, max_steps=20, surface_model="fresnel")
    torch.cuda.synchronize()
    ms_pt = (time.perf_counter() - t0) * 1000.0

    tw = WarpTracer(verts, faces, fm, mats, device)
    tw.trace(rays, max_steps=2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tw.trace(rays, max_steps=20)
    torch.cuda.synchronize()
    ms_wp = (time.perf_counter() - t0) * 1000.0

    speedup = ms_pt / ms_wp if ms_wp > 0 else float("inf")

    print("\n[Test 1] Warp speedup regression")
    print("  PyTorch: {:.2f} ms".format(ms_pt))
    print("  Warp   : {:.3f} ms".format(ms_wp))
    print("  Speedup: {:.1f}x".format(speedup))

    assert speedup > 5.0


@_skip_warp
def test_warp_memory_budget():
    device = torch.device("cuda")
    verts, faces, fm, mats = make_scene(device, n_grid=4)
    rays = make_rays(device, N=65536)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tw = WarpTracer(verts, faces, fm, mats, device)
    try:
        out = tw.trace(rays, max_steps=20)
    except torch.cuda.OutOfMemoryError:
        pytest.skip("Out of memory even at 65536 rays")

    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    print("\n[Test 2] Warp memory budget (N=65536, 20 steps)")
    print("  Peak VRAM: {:.1f} MB".format(peak_mb))

    assert peak_mb < 2000


@_skip_warp
def test_warp_large_batch_scaling():
    device = torch.device("cuda")
    verts, faces, fm, mats = make_scene(device, n_grid=4)

    tw = WarpTracer(verts, faces, fm, mats, device)

    timings = {}
    for N in [1024, 16384]:
        rays = make_rays(device, N)
        tw.trace(rays, max_steps=2)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tw.trace(rays, max_steps=20)
        torch.cuda.synchronize()
        timings[N] = (time.perf_counter() - t0) * 1000.0

    ratio = timings[16384] / timings[1024] if timings[1024] > 0 else float("inf")

    print("\n[Test 3] Warp large-batch scaling")
    print("  N=1024 : {:.3f} ms".format(timings[1024]))
    print("  N=16384: {:.3f} ms".format(timings[16384]))
    print("  Ratio  : {:.2f}x".format(ratio))

    assert ratio < 16.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))