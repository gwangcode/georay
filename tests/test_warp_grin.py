"""
tests/test_warp_grin.py
========================
WarpTracer 的 GRIN 功能完整测试。

Run:
    pytest tests/test_warp_grin.py -v -s
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, WarpTracer, generate_rays


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
    box = trimesh.creation.box(extents=list(extents))
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def make_parabolic_grin_materials(n0=1.5, A=0.1):
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
    PyTorch 用 callable，Warp 用 parabolic dict。
    """
    import numpy as np

    device = torch.device("cuda")
    n0, A = 1.5, 0.1

    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 4.0))

    def parabolic_grin(x, y, z, wavelength):
        return n0 - A * (x ** 2 + y ** 2)

    materials_pt = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": n0,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": parabolic_grin,
        },
    }

    materials_wp = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": n0,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": {"type": "parabolic", "n0": n0, "A": A},
        },
    }

    rays = make_rays_z(device, n=16, x_bounds=(-0.3, 0.3), y_bounds=(-0.3, 0.3))

    tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials_pt, device)
    out_pt = tp.trace(rays, max_steps=200, ds_init=0.02,
                      ds_min=0.02, ds_max=0.02, rk_tol=1e9,
                      fresnel=False)

    tw = WarpTracer(verts, faces, fm, materials_wp, device)
    out_wp = tw.trace(rays, max_steps=200, ds_init=0.02)

    S_min = min(out_pt["positions"].shape[1], out_wp["positions"].shape[1])

    print("\n[Test 1] Warp vs PyTorch GRIN Parabolic")
    print("  PyTorch steps: {}".format(out_pt["positions"].shape[1]))
    print("  Warp steps   : {}".format(out_wp["positions"].shape[1]))

    pos_diff = (out_pt["positions"][:, :S_min] -
                out_wp["positions"][:, :S_min]).abs().max().item()
    print("  max |Delta positions|: {:.3e}".format(pos_diff))

    assert pos_diff < 1e-2, \
        "Warp vs PyTorch parabolic GRIN mismatch: max |Delta| = {:.6e}".format(pos_diff)


# ==============================================================================
# Test 2: 抛物线 GRIN 透镜聚焦
# ==============================================================================

@_skip_warp
def test_warp_grin_parabolic_lens_focus():
    device = torch.device("cuda")
    n0, A = 1.5, 0.15
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 2.0))
    materials = make_parabolic_grin_materials(n0=n0, A=A)

    rays = make_rays_z(device, n=16, x_bounds=(-0.4, 0.4), y_bounds=(-0.4, 0.4))

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=400, ds_init=0.01)

    pos = out["positions"]
    x_start = pos[:, 0, 0]
    x_end = pos[:, -1, 0]

    significant = x_start.abs() > 0.1
    x_start_sig = x_start[significant].abs().mean().item()
    x_end_sig = x_end[significant].abs().mean().item()

    print("\n[Test 2] Parabolic GRIN Lens Focusing")
    print("  n0={}, A={}, L=2.0".format(n0, A))
    print("  Mean |x_start| = {:.4f}".format(x_start_sig))
    print("  Mean |x_end|   = {:.4f}".format(x_end_sig))
    print("  Bend ratio = {:.3f}".format(
        x_end_sig / x_start_sig if x_start_sig > 0 else 0))

    assert x_end_sig < 0.9 * x_start_sig, \
        "GRIN lens did not bend rays: x_start={:.4f}, x_end={:.4f}".format(
            x_start_sig, x_end_sig)


# ==============================================================================
# Test 3: 3D grid 线性场
# ==============================================================================

@_skip_warp
def test_warp_grin_3d_grid_linear_field():
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

    pos = out["positions"][0]
    x_start = pos[0, 0].item()
    x_end = pos[-1, 0].item()

    print("\n[Test 3] 3D Grid Linear Field n = 1.5 + 0.1x")
    print("  x_start = {:.4f}, x_end = {:.4f}, delta = {:.4f}".format(
        x_start, x_end, x_end - x_start))

    assert x_end > x_start + 0.05, \
        "GRIN 3D grid: ray did not bend toward +x"


# ==============================================================================
# Test 4: 3D grid 梯度方向
# ==============================================================================

@_skip_warp
def test_warp_grin_3d_grid_gradient_direction():
    import numpy as np

    device = torch.device("cuda")

    Nx, Ny, Nz = 32, 4, 4
    x = np.linspace(-1.0, 1.0, Nx).astype(np.float32)
    y = np.linspace(-1.0, 1.0, Ny).astype(np.float32)
    z = np.linspace(-2.0, 2.0, Nz).astype(np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
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

    x_starts = [-0.5, 0.0, 0.5]

    print("\n[Test 4] 3D Grid Gradient Direction (n = 1.5 + 0.2x)")
    print("  x_start   x_end     delta_x")
    print("  " + "-" * 35)

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

        print("  {:+.3f}    {:+.4f}   {:+.4f}".format(xs, xe, dx))

        assert dx > 0.0, \
            "Ray starting at x={} did not bend toward +x: delta_x={:.4f}".format(
                x0, dx)


# ==============================================================================
# Test 5: 4D grid 色散
# ==============================================================================

@_skip_warp
def test_warp_grin_4d_wavelength_dispersion():
    import numpy as np

    device = torch.device("cuda")

    Nx, Ny, Nz, Nw = 16, 4, 4, 4
    x = np.linspace(-1.0, 1.0, Nx).astype(np.float32)
    y = np.linspace(-1.0, 1.0, Ny).astype(np.float32)
    z = np.linspace(-2.0, 2.0, Nz).astype(np.float32)
    wl = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32)

    X, Y, Z, W = np.meshgrid(x, y, z, wl, indexing="ij")
    n_base = 1.55 - 0.0001 * (W - 400.0)
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
    print("  lambda   x_start   x_end")
    print("  " + "-" * 35)

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
        print("  {:.0f}nm   {:+.4f}    {:+.4f}".format(w, xs, xe))

    diff = abs(x_finals[400.0] - x_finals[700.0])
    print("  |x_end(400) - x_end(700)| = {:.4f}".format(diff))
    assert diff > 1e-4, \
        "4D GRIN dispersion: delta = {:.6e}".format(diff)


# ==============================================================================
# Test 6: callable GRIN（采样成 3D grid）
# ==============================================================================

@_skip_warp
def test_warp_grin_callable():
    """
    callable GRIN：用户提供的 Python 函数被自动采样成 3D grid，
    之后走 grid 路径（三线性插值 + 解析梯度）。
    """
    import numpy as np

    device = torch.device("cuda")
    verts, faces, fm = make_grin_box(device, extents=(2.0, 2.0, 4.0))

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": 1.5,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": {
                "type": "callable",
                "bounds": (-1.0, 1.0, -1.0, 1.0, -2.0, 2.0),
                "resolution": (32, 32, 32),
            },
        },
    }

    def my_linear_grin(x, y, z):
        return 1.5 + 0.1 * x

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

    print("\n[Test 6] Callable GRIN (auto-sampled to 3D grid)")
    print("  n(x) = 1.5 + 0.1x")
    print("  Grid bounds: (-1, 1, -1, 1, -2, 2), resolution: 32^3")
    print("  x_start = {:.4f}, x_end = {:.4f}, delta = {:.4f}".format(
        x_start, x_end, x_end - x_start))

    assert x_end > x_start + 0.01, \
        "Callable GRIN: ray did not bend toward +x"


# ==============================================================================
# Test 7: 能量守恒
# ==============================================================================

@_skip_warp
def test_warp_grin_energy_conserved():
    device = torch.device("cuda")
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 4.0))
    materials = make_parabolic_grin_materials(n0=1.5, A=0.1)

    rays = make_rays_z(device, n=16)

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=200, ds_init=0.02)

    energy = out["energy"]

    print("\n[Test 7] GRIN Energy Conservation")
    print("  Energy range: [{:.6f}, {:.6f}]".format(
        energy.min().item(), energy.max().item()))

    assert torch.allclose(energy, torch.ones_like(energy), atol=1e-5), \
        "GRIN stepping changed energy: max |E-1| = {:.3e}".format(
            (energy - 1.0).abs().max().item())

    assert out["energy_conserved"], "energy_conserved flag is False"


# ==============================================================================
# Test 8: 性能基准
# ==============================================================================

@_skip_warp
def test_warp_grin_performance_benchmark():
    import time

    device = torch.device("cuda")
    n0, A = 1.5, 0.1
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 4.0))

    def parabolic_grin(x, y, z, wavelength):
        return n0 - A * (x ** 2 + y ** 2)

    materials_pt = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": n0,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": parabolic_grin,
        },
    }
    materials_wp = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": n0,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": {"type": "parabolic", "n0": n0, "A": A},
        },
    }

    print("\n[Test 8] GRIN Performance Benchmark")
    print("  {:>8s} {:>18s} {:>18s} {:>14s}".format(
        "N_rays", "PyTorch (ms)", "Warp (ms)", "Speedup"))
    print("  " + "-" * 62)

    for N in [256, 1024, 4096]:
        rays = make_rays_z(device, n=N)

        tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials_pt, device)
        tp.trace(rays, max_steps=100, ds_init=0.02, fresnel=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tp.trace(rays, max_steps=100, ds_init=0.02, fresnel=False)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) * 1000

        tw = WarpTracer(verts, faces, fm, materials_wp, device)
        tw.trace(rays, max_steps=100, ds_init=0.02)
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
    RK4 步长收敛：随着 ds 减小，x_end 应收敛到稳定值。
    """
    device = torch.device("cuda")
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 4.0))
    materials = make_parabolic_grin_materials(n0=1.5, A=0.1)

    rays = make_rays_z(device, n=1, x_bounds=(0.3, 0.3), y_bounds=(0.0, 0.0))

    tw = WarpTracer(verts, faces, fm, materials, device)

    print("\n[Test 9] RK4 Step Size Convergence")
    print("  {:>8s} {:>14s}".format("ds", "x_end"))
    print("  " + "-" * 26)

    ds_values = [0.08, 0.04, 0.02, 0.01, 0.005]
    x_ends = []
    for ds in ds_values:
        out = tw.trace(rays, max_steps=4000, ds_init=ds)
        x_end = out["positions"][0, -1, 0].item()
        x_ends.append(x_end)
        print("  {:>8.4f} {:>14.6f}".format(ds, x_end))

    diff_final = abs(x_ends[-1] - x_ends[-2])
    print("  |x_end(0.005) - x_end(0.01)| = {:.3e}".format(diff_final))
    assert diff_final < 1e-3, \
        "RK4 not converging: diff = {:.6e}".format(diff_final)

    x_start = 0.3
    for ds, x_end in zip(ds_values, x_ends):
        assert abs(x_end) < abs(x_start), \
            "ds={:.4f}: |x_end|={:.4f} not < |x_start|={:.4f}".format(
                ds, abs(x_end), abs(x_start))

    print("  -> RK4 converged to x_end = {:.6f}".format(x_ends[-1]))


# ==============================================================================
# Test 10: 边界行为
# ==============================================================================

@_skip_warp
def test_warp_grin_boundary_behavior():
    import numpy as np

    device = torch.device("cuda")

    Nx, Ny, Nz = 8, 4, 4
    x = np.linspace(-0.5, 0.5, Nx).astype(np.float32)
    y = np.linspace(-0.5, 0.5, Ny).astype(np.float32)
    z = np.linspace(-1.0, 1.0, Nz).astype(np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
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
    print("  Ray starts at x = -1.5 (outside grid)")
    print("  x_end = {:.4f}".format(pos[-1, 0].item()))
    print("  energy range: [{:.6f}, {:.6f}]".format(
        energy.min().item(), energy.max().item()))

    assert not torch.isnan(pos).any(), "Boundary GRIN produced NaN"
    assert not torch.isnan(energy).any(), "Boundary GRIN produced NaN energy"
    assert torch.allclose(energy, torch.ones_like(energy), atol=1e-5), \
        "Boundary GRIN broke energy conservation"

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