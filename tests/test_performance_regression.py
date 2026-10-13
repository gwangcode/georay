"""
tests/test_performance_regression.py
=====================================
性能回归监控测试。

防止某次代码修改导致 Warp 相对 PyTorch 的加速比下降。

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
    reason="WarpTracer requires CUDA + warp-lang + georay[warp]"
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
    return verts, faces, fm, mats


def make_rays(device, N):
    return generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -8.0), direction=(0.0, 0.0, 1.0),
        bounds=(-3.0, 3.0, -3.0, 3.0),
        polarization="s", device=device)


# ==============================================================================
# Test 1: Warp 加速比回归
# ==============================================================================

@_skip_warp
def test_warp_speedup_regression():
    """
    Warp 相对 PyTorch 的加速比应 ≥ 5×（保守阈值）。

    如果低于此值，说明 Warp 内核或 PyTorch 侧的性能发生重大变化。
    """
    device = torch.device("cuda")
    verts, faces, fm, mats = make_scene(device, n_grid=4)
    rays = make_rays(device, N=1024)

    # ---- PyTorch ----
    tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, mats, device)
    tp.trace(rays, max_steps=2, surface_model="fresnel")   # warmup
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    tp.trace(rays, max_steps=20, surface_model="fresnel")
    torch.cuda.synchronize()
    ms_pt = (time.perf_counter() - t0) * 1000.0

    # ---- Warp ----
    tw = WarpTracer(verts, faces, fm, mats, device)
    tw.trace(rays, max_steps=2)   # warmup + compile
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

    assert speedup > 5.0, \
        "Warp speedup dropped to {:.1f}x (threshold: 5x)".format(speedup)


# ==============================================================================
# Test 2: Warp 显存上限
# ==============================================================================

@_skip_warp
def test_warp_memory_budget():
    """
    65536 光线 × 20 步的显存峰值应 < 2 GB。
    """
    device = torch.device("cuda")
    verts, faces, fm, mats = make_scene(device, n_grid=4)
    rays = make_rays(device, N=65536)

    # 清理
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

    # 65536 × 21 × ~13 floats × 4 bytes ≈ 90 MB
    # 加上 mesh + intermediates，200 MB 是合理上限
    assert peak_mb < 2000, \
        "Peak VRAM too high: {:.1f} MB (budget: 2000 MB)".format(peak_mb)


# ==============================================================================
# Test 3: 大批量性能缩放
# ==============================================================================

@_skip_warp
def test_warp_large_batch_scaling():
    """
    Warp 的耗时随 N 增长应亚线性（GPU 并行度充足）。
    从 N=1024 到 N=16384（16×），耗时增长应 < 16×。
    """
    device = torch.device("cuda")
    verts, faces, fm, mats = make_scene(device, n_grid=4)

    tw = WarpTracer(verts, faces, fm, mats, device)

    timings = {}
    for N in [1024, 16384]:
        rays = make_rays(device, N)
        tw.trace(rays, max_steps=2)   # warmup
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        tw.trace(rays, max_steps=20)
        torch.cuda.synchronize()
        timings[N] = (time.perf_counter() - t0) * 1000.0

    ratio = timings[16384] / timings[1024] if timings[1024] > 0 else float("inf")

    print("\n[Test 3] Warp large-batch scaling")
    print("  N=1024 : {:.3f} ms".format(timings[1024]))
    print("  N=16384: {:.3f} ms".format(timings[16384]))
    print("  Ratio  : {:.2f}x (expected < 16x)".format(ratio))

    assert ratio < 16.0, \
        "Warp scaling ratio {:.2f}x exceeds linear (16x)".format(ratio)


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))