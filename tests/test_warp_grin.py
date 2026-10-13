"""
tests/test_warp_grin.py
========================
WarpTracer 的 GRIN 功能完整测试。

覆盖：
    1. Warp vs PyTorch：抛物线 GRIN 数值一致性
    2. 抛物线 GRIN 透镜聚焦
    3. 3D grid：线性场插值
    4. 3D grid：梯度方向
    5. 4D grid：色散
    6. callable GRIN：@wp.func 桥接
    7. GRIN 步进的能量守恒
    8. 性能基准（Warp vs PyTorch）
    9. RK4 步长收敛性
   10. GRIN 场边界行为

Run:
    pytest tests/test_warp_grin.py -v -s
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, WarpTracer, generate_rays


# ==============================================================================
# 环境守卫
# ==============================================================================

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


# ==============================================================================
# 场景构造
# ==============================================================================

def make_grin_box(device, extents=(1.0, 1.0, 4.0)):
    """GRIN 介质盒。"""
    box = trimesh.creation.box(extents=list(extents))
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0   # 外部 = air
    fm[:, 1] = 1   # 内部 = GRIN 介质
    return verts, faces, fm


def make_parabolic_grin_materials(n0=1.5, A=0.1, wl_grid=None):
    """抛物线 GRIN 材质。"""
    mat = {
        "n": n0,
        "reflectivity": 0.0,
        "diffuse": 0.0,
        "absorptivity": 0.0,
        "grin_field": {"type": "parabolic", "n0": n0, "A": A},
    }
    return {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: mat,
    }


def make_rays_z(device, n=4, x_bounds=(-0.3, 0.3), y_bounds=(-0.3, 0.3),
                z_start=-3.0, wavelength_nm=550.0):
    return generate_rays(
        source_type="grid", num_rays=n,
        origin=(0.0, 0.0, z_start), direction=(0.0, 0.0, 1.0),
        bounds=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]),
        polarization="s", device=device, wavelength_nm=wavelength_nm)


# ==============================================================================
# Test 1: Warp vs PyTorch 抛物线 GRIN 数值一致性
# ==============================================================================

@_skip_warp
def test_warp_grin_parabolic_matches_pytorch():
    """
    抛物线 GRIN：Warp 与 PyTorch 追踪应数值一致。

    为保证公平对比：
      - PyTorch 用固定步长（rk_tol=1e9 强制接受）
      - 两边用相同 ds_init / max_steps / 光线起点
    """
    device = torch.device("cuda")
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 4.0))
    materials = make_parabolic_grin_materials(n0=1.5, A=0.1)

    rays = make_rays_z(device, n=16, x_bounds=(-0.3, 0.3), y_bounds=(-0.3, 0.3))

    # ---- PyTorch ----
    tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out_pt = tp.trace(rays, max_steps=200, ds_init=0.02,
                      ds_min=0.02, ds_max=0.02, rk_tol=1e9,
                      fresnel=False)

    # ---- Warp ----
    tw = WarpTracer(verts, faces, fm, materials, device)
    out_wp = tw.trace(rays, max_steps=200, ds_init=0.02)

    S_min = min(out_pt["positions"].shape[1], out_wp["positions"].shape[1])

    print("\n[Test 1] Warp vs PyTorch GRIN Parabolic")
    print("  PyTorch steps: {}".format(out_pt["positions"].shape[1]))
    print("  Warp steps   : {}".format(out_wp["positions"].shape[1]))

    pos_diff = (out_pt["positions"][:, :S_min] -
                out_wp["positions"][:, :S_min]).abs().max().item()
    print("  max |Delta positions| (common prefix): {:.3e}".format(pos_diff))

    # 两边都用相同 RK4，理论上应完全一致
    assert pos_diff < 1e-3, \
        "Warp vs PyTorch parabolic GRIN mismatch: max |Delta| = {:.6e}".format(pos_diff)


# ==============================================================================
# Test 2: 抛物线 GRIN 透镜聚焦
# ==============================================================================

@_skip_warp
def test_warp_grin_parabolic_lens_focus():
    """
    抛物线 GRIN（n = n0 - A·(x² + y²)）是梯度折射率透镜。
    离轴光线应向光轴弯曲，|x_final| < |x_initial|。

    物理近似：GRIN 薄透镜焦距 f ≈ n0 / (2·A·L)。
    对 n0=1.5, A=0.15, L=2.0：f ≈ 2.5。
    """
    device = torch.device("cuda")
    n0, A = 1.5, 0.15
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 2.0))
    materials = make_parabolic_grin_materials(n0=n0, A=A)

    # 起点在 z=-3，光线沿 +z 进入 z=±1 的介质，离轴 x = ±0.3
    rays = make_rays_z(device, n=16, x_bounds=(-0.4, 0.4), y_bounds=(-0.4, 0.4))

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=400, ds_init=0.01)

    pos = out["positions"]
    x_start = pos[:, 0, 0]
    x_end = pos[:, -1, 0]

    # 只看有明显离轴的光线
    significant = x_start.abs() > 0.1
    x_start_sig = x_start[significant].abs().mean().item()
    x_end_sig = x_end[significant].abs().mean().item()

    print("\n[Test 2] Parabolic GRIN Lens Focusing")
    print("  n0={}, A={}, L=2.0".format(n0, A))
    print("  Analytic focal length f ~ n0/(2*A*L) = {:.3f}".format(n0 / (2 * A * 2.0)))
    print("  Mean |x_start| (significant rays) = {:.4f}".format(x_start_sig))
    print("  Mean |x_end|   (significant rays) = {:.4f}".format(x_end_sig))
    print("  Bend ratio |x_end| / |x_start| = {:.3f}".format(
        x_end_sig / x_start_sig if x_start_sig > 0 else 0))

    # 必须明显向轴弯曲
    assert x_end_sig < 0.9 * x_start_sig, \
        "GRIN lens did not bend rays enough: x_start={:.4f}, x_end={:.4f}".format(
            x_start_sig, x_end_sig)


# ==============================================================================
# Test 3: 3D grid 线性场插值
# ==============================================================================

@_skip_warp
def test_warp_grin_3d_grid_linear_field():
    """
    3D grid GRIN：n(x, y, z) = 1.5 + 0.1·x

    光线沿 +z 走，在均匀场梯度下应向 +x 弯曲。
    """
    import numpy as np

    device = torch.device("cuda")

    Nx, Ny, Nz = 16, 4, 4
    x = np.linspace(-1.0, 1.0, Nx).astype(np.float32)
    y = np.linspace(-1.0, 1.0, Ny).astype(np.float32)
    z = np.linspace(-2.0, 2.0, Nz).astype(np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    n_values = 1.5 + 0.1 * X

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": 1.5,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": {
                "type": "grid",
                "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
                "values": n_values.tolist(),
            },
        },
    }

    verts, faces, fm = make_grin_box(device, extents=(2.0, 2.0, 4.0))

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=400, ds_init=0.02)

    pos = out["positions"][0]  # [S+1, 3]
    x_start = pos[0, 0].item()
    x_end = pos[-1, 0].item()

    print("\n[Test 3] 3D Grid Linear Field n = 1.5 + 0.1x")
    print("  Ray starts at x = {:.4f}".format(x_start))
    print("  Ray ends   at x = {:.4f}".format(x_end))
    print("  Delta x = {:.4f}".format(x_end - x_start))

    # 梯度 (0.1, 0, 0) 使光线向 +x 弯曲
    assert x_end > x_start + 0.05, \
        "GRIN 3D grid linear field: ray did not bend toward +x: " \
        "x_start={:.4f}, x_end={:.4f}".format(x_start, x_end)


# ==============================================================================
# Test 4: 3D grid 梯度方向
# ==============================================================================

@_skip_warp
def test_warp_grin_3d_grid_gradient_direction():
    """
    3D grid GRIN：n(x) = 1.5 + 0.2·x  (梯度沿 +x)

    从不同起点出发的光线都应向 +x 弯曲。
    如果梯度方向搞反，光线会向 -x 弯曲。
    """
    import numpy as np

    device = torch.device("cuda")

    Nx, Ny, Nz = 32, 4, 4
    x = np.linspace(-1.0, 1.0, Nx).astype(np.float32)
    y = np.linspace(-1.0, 1.0, Ny).astype(np.float32)
    z = np.linspace(-2.0, 2.0, Nz).astype(np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # 强梯度：n = 1.5 + 0.2x
    n_values = 1.5 + 0.2 * X

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": 1.5,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": {
                "type": "grid",
                "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
                "values": n_values.tolist(),
            },
        },
    }

    verts, faces, fm = make_grin_box(device, extents=(2.0, 2.0, 4.0))

    # 从 x = -0.5, 0, +0.5 出发
    x_starts = [-0.5, 0.0, 0.5]

    print("\n[Test 4] 3D Grid Gradient Direction (n = 1.5 + 0.2x)")
    print("  x_start   x_end     delta_x    bend_toward")
    print("  " + "-" * 45)

    for x0 in x_starts:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(x0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device)

        tw = WarpTracer(verts, faces, fm, materials, device)
        out = tw.trace(rays, max_steps=400, ds_init=0.01)

        pos = out["positions"][0]
        xs = pos[0, 0].item()
        xe = pos[-1, 0].item()
        dx = xe - xs
        bend = "+x" if dx > 0.01 else ("-x" if dx < -0.01 else "none")

        print("  {:+.3f}    {:+.4f}   {:+.4f}   {}".format(xs, xe, dx, bend))

        # 梯度恒为 +0.2x，光线应向 +x 弯曲
        assert dx > 0.0, \
            "Ray starting at x={} did not bend toward +x: delta_x={:.4f}".format(
                x0, dx)


# ==============================================================================
# Test 5: 4D grid 色散
# ==============================================================================

@_skip_warp
def test_warp_grin_4d_wavelength_dispersion():
    """
    4D GRIN grid：n(x, λ) = n_ref(λ) - A·x²

    不同波长的折射率基准不同 → 聚焦强度不同 → 末端 |x| 不同。
    """
    import numpy as np

    device = torch.device("cuda")

    Nx, Ny, Nz, Nw = 16, 4, 4, 4
    x = np.linspace(-1.0, 1.0, Nx).astype(np.float32)
    y = np.linspace(-1.0, 1.0, Ny).astype(np.float32)
    z = np.linspace(-2.0, 2.0, Nz).astype(np.float32)
    wl = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32)

    X, Y, Z, W = np.meshgrid(x, y, z, wl, indexing="ij")
    # 基准折射率随波长变化（色散）
    n_base = 1.55 - 0.0001 * (W - 400.0)
    # 抛物线弯曲
    n_values = n_base - 0.15 * (X ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": 1.5,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": {
                "type": "grid_4d",
                "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
                "wavelengths_nm": wl.tolist(),
                "values": n_values.tolist(),
            },
        },
    }

    verts, faces, fm = make_grin_box(device, extents=(2.0, 2.0, 4.0))

    x_finals = {}
    print("\n[Test 5] 4D Grid Wavelength Dispersion")
    print("  lambda   x_start   x_end     |x_end|")
    print("  " + "-" * 45)

    for w in [400.0, 700.0]:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.5, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=w)

        tw = WarpTracer(verts, faces, fm, materials, device)
        out = tw.trace(rays, max_steps=400, ds_init=0.02)

        pos = out["positions"][0]
        xs = pos[0, 0].item()
        xe = pos[-1, 0].item()
        x_finals[w] = abs(xe)

        print("  {:.0f}nm   {:+.4f}    {:+.4f}     {:.4f}".format(
            w, xs, xe, abs(xe)))

    # 两波长结果应不同（色散）
    diff = abs(x_finals[400.0] - x_finals[700.0])
    print("  |x_end(400nm) - x_end(700nm)| = {:.4f}".format(diff))

    assert diff > 1e-4, \
        "4D GRIN should show wavelength dispersion, but delta = {:.6e}".format(diff)


# ==============================================================================
# Test 6: callable GRIN
# ==============================================================================

@_skip_warp
def test_warp_grin_callable():
    """
    callable GRIN：用户提供 @wp.func，编译期内联到内核。

    测试一个简单的线性场 n = 1.5 + 0.1·x，
    光线应向 +x 弯曲。
    """
    import warp as wp

    # 用户提供的 @wp.func
    @wp.func
    def my_linear_grin(x: wp.float32, y: wp.float32, z: wp.float32):
        return 1.5 + 0.1 * x

    device = torch.device("cuda")
    verts, faces, fm = make_grin_box(device, extents=(2.0, 2.0, 4.0))

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": 1.5,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": {"type": "callable"},
        },
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tw = WarpTracer(verts, faces, fm, materials, device,
                    grin_callable=my_linear_grin)
    out = tw.trace(rays, max_steps=400, ds_init=0.02)

    pos = out["positions"][0]
    x_start = pos[0, 0].item()
    x_end = pos[-1, 0].item()

    print("\n[Test 6] Callable GRIN")
    print("  n(x) = 1.5 + 0.1x")
    print("  x_start = {:.4f}, x_end = {:.4f}, delta_x = {:.4f}".format(
        x_start, x_end, x_end - x_start))

    assert x_end > x_start + 0.01, \
        "Callable GRIN: ray did not bend toward +x: " \
        "x_start={:.4f}, x_end={:.4f}".format(x_start, x_end)


# ==============================================================================
# Test 7: GRIN 步进的能量守恒
# ==============================================================================

@_skip_warp
def test_warp_grin_energy_conserved():
    """
    GRIN 步进是几何操作，不应改变能量。
    场景中若无界面交互（光线始终在 GRIN 介质内），
    能量应保持 1.0。
    """
    device = torch.device("cuda")
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 4.0))
    materials = make_parabolic_grin_materials(n0=1.5, A=0.1)

    rays = make_rays_z(device, n=16)

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=200, ds_init=0.02)

    energy = out["energy"]  # [N, S+1]

    print("\n[Test 7] GRIN Energy Conservation")
    print("  Energy range: [{:.6f}, {:.6f}]".format(
        energy.min().item(), energy.max().item()))
    print("  Max deviation from 1.0: {:.3e}".format(
        (energy - 1.0).abs().max().item()))

    # GRIN 步进不应改变能量
    assert torch.allclose(energy, torch.ones_like(energy), atol=1e-5), \
        "GRIN stepping changed energy: max |E-1| = {:.3e}".format(
            (energy - 1.0).abs().max().item())

    assert out["energy_conserved"], "energy_conserved flag is False"


# ==============================================================================
# Test 8: 性能基准
# ==============================================================================

@_skip_warp
def test_warp_grin_performance_benchmark():
    """WarpTracer GRIN vs PyTorch GRIN 性能对比。"""
    import time

    device = torch.device("cuda")
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 4.0))
    materials = make_parabolic_grin_materials(n0=1.5, A=0.1)

    print("\n[Test 8] GRIN Performance Benchmark")
    print("  {:>8s} {:>18s} {:>18s} {:>14s}".format(
        "N_rays", "PyTorch (ms)", "Warp (ms)", "Speedup"))
    print("  " + "-" * 62)

    for N in [256, 1024, 4096]:
        rays = make_rays_z(device, n=N)

        # PyTorch
        tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        tp.trace(rays, max_steps=100, ds_init=0.02, fresnel=False)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tp.trace(rays, max_steps=100, ds_init=0.02, fresnel=False)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) * 1000

        # Warp
        tw = WarpTracer(verts, faces, fm, materials, device)
        tw.trace(rays, max_steps=100, ds_init=0.02)  # warmup (compiles)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tw.trace(rays, max_steps=100, ds_init=0.02)
        torch.cuda.synchronize()
        t_wp = (time.perf_counter() - t0) * 1000

        print("  {:>8d} {:>18.2f} {:>18.3f} {:>14.1f}x".format(
            N, t_pt, t_wp, t_pt / t_wp if t_wp > 0 else float("inf")))


# ==============================================================================
# Test 9: RK4 步长收敛性
# ==============================================================================

@_skip_warp
def test_warp_grin_step_size_convergence():
    """
    RK4 步长收敛：ds 减半，误差应下降约 16 倍（4 阶方法）。

    用同一个抛物线 GRIN 场景，跑不同 ds，比较最终位置。
    基准用极小 ds（近似解析解）。
    """
    device = torch.device("cuda")
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 4.0))
    materials = make_parabolic_grin_materials(n0=1.5, A=0.1)

    rays = make_rays_z(device, n=1, x_bounds=(0.3, 0.3), y_bounds=(0.0, 0.0))

    # 基准（极小步长）
    tw = WarpTracer(verts, faces, fm, materials, device)
    out_ref = tw.trace(rays, max_steps=2000, ds_init=0.002)
    x_ref = out_ref["positions"][0, -1, 0].item()

    print("\n[Test 9] RK4 Step Size Convergence")
    print("  Reference (ds=0.002): x_end = {:.6f}".format(x_ref))
    print("  {:>8s} {:>14s} {:>14s} {:>10s}".format(
        "ds", "x_end", "|err|", "ratio"))
    print("  " + "-" * 50)

    errors = []
    for ds in [0.08, 0.04, 0.02, 0.01]:
        out = tw.trace(rays, max_steps=2000, ds_init=ds)
        x_end = out["positions"][0, -1, 0].item()
        err = abs(x_end - x_ref)
        errors.append(err)
        ratio = errors[-2] / err if len(errors) > 1 and err > 0 else 0
        print("  {:>8.4f} {:>14.6f} {:>14.3e} {:>10.2f}".format(
            ds, x_end, err, ratio))

    # RK4 是 4 阶：ds 减半，误差应下降至少 4 倍（保守估计）
    # 精确值 ≈ 16，但数值噪声会降低实际比率
    if len(errors) >= 2 and errors[-1] > 0 and errors[-2] > 0:
        ratio = errors[-2] / errors[-1]
        print("  Final convergence ratio (ds 0.02 -> 0.01): {:.2f}x".format(ratio))
        assert ratio > 3.0, \
            "RK4 convergence too slow: ratio = {:.2f} (expected > 3)".format(ratio)


# ==============================================================================
# Test 10: GRIN 场边界行为
# ==============================================================================

@_skip_warp
def test_warp_grin_boundary_behavior():
    """
    GRIN 场网格外的行为应明确：
      * x < x_min: n = n(x_min)  (clamp)
      * x > x_max: n = n(x_max)  (clamp)

    不应产生 NaN 或极端值。
    """
    import numpy as np

    device = torch.device("cuda")

    Nx, Ny, Nz = 8, 4, 4
    x = np.linspace(-0.5, 0.5, Nx).astype(np.float32)
    y = np.linspace(-0.5, 0.5, Ny).astype(np.float32)
    z = np.linspace(-1.0, 1.0, Nz).astype(np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # 线性场：n = 1.5 + 0.2x
    # x = -0.5 → n = 1.4, x = +0.5 → n = 1.6
    n_values = 1.5 + 0.2 * X

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": 1.5,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": {
                "type": "grid",
                "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
                "values": n_values.tolist(),
            },
        },
    }

    verts, faces, fm = make_grin_box(device, extents=(4.0, 4.0, 4.0))

    # 光线起点 x = -1.5（远超网格 x_min = -0.5）
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(-1.5, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=200, ds_init=0.02)

    pos = out["positions"][0]
    energy = out["energy"][0]

    print("\n[Test 10] GRIN Boundary Behavior")
    print("  Grid x range: [-0.5, +0.5]")
    print("  Ray starts at x = -1.5 (outside grid, x_min side)")
    print("  x_end = {:.4f}".format(pos[-1, 0].item()))
    print("  energy range: [{:.6f}, {:.6f}]".format(
        energy.min().item(), energy.max().item()))

    # 检查无 NaN
    assert not torch.isnan(pos).any(), "Boundary GRIN produced NaN positions"
    assert not torch.isnan(energy).any(), "Boundary GRIN produced NaN energy"

    # 检查能量守恒
    assert torch.allclose(energy, torch.ones_like(energy), atol=1e-5), \
        "Boundary GRIN broke energy conservation"

    # 光线向 +x 弯曲（因为 clamp 后 n 仍是线性递增的）
    x_start = pos[0, 0].item()
    x_end = pos[-1, 0].item()
    print("  delta_x = {:.4f}".format(x_end - x_start))
    assert x_end > x_start, \
        "Boundary GRIN: ray did not bend toward +x"


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))