"""
tests/test_warp_equivalence.py
===============================
WarpTracer 与 PyTorch tracer 在随机分支和 GRIN 场景下的一致性。
"""
import math
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
    from georay import WarpTracer
    _HAS_WARP_TRACER = WarpTracer is not None
except ImportError:
    WarpTracer = None
    _HAS_WARP_TRACER = False


_skip_warp = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_WARP and _HAS_WARP_TRACER),
    reason="WarpTracer requires CUDA + warp-lang + georay[warp]"
)


def make_flat_interface(device, hs=100.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def make_grin_box(device, extents=(1.0, 1.0, 2.0)):
    box = trimesh.creation.box(extents=list(extents))
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


# ==============================================================================
# Test 1: Warp vs PyTorch 漫反射统计一致性
# ==============================================================================

@_skip_warp
def test_warp_diffuse_statistical_equivalence():
    """
    WarpTracer 与 PyTorch 的漫反射半球采样应统计等价。
    """
    device = torch.device("cuda")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 1.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 1.0, "absorptivity": 0.0},
    }

    N = 1024
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.01, 0.01, -0.01, 0.01),
        polarization="s", device=device)

    def mean_cos(out):
        p1 = out["positions"][:, 1]
        p2 = out["positions"][:, 2]
        active = out["active"][:, 1]
        d = p2 - p1
        norms = d.norm(dim=-1, keepdim=True)
        valid = active & (norms.squeeze(-1) > 1e-3)
        d_unit = d[valid] / (norms[valid] + 1e-12)
        return (-d_unit[:, 2]).mean().item()

    tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out_pt = tp.trace(rays, max_steps=3, surface_model="custom")

    tw = WarpTracer(verts, faces, fm, materials, device)
    out_wp = tw.trace(rays, max_steps=3, surface_model="custom", seed=42)

    m_pt = mean_cos(out_pt)
    m_wp = mean_cos(out_wp)

    print("\n[Test 1] Warp vs PyTorch diffuse")
    print("  PyTorch mean cos θ: {:.4f}".format(m_pt))
    print("  Warp    mean cos θ: {:.4f}".format(m_wp))

    assert abs(m_pt - 2.0/3.0) < 0.05
    assert abs(m_wp - 2.0/3.0) < 0.05


# ==============================================================================
# Test 2: Warp vs PyTorch GRIN 3D grid 一致性
# ==============================================================================

@_skip_warp
def test_warp_grin_3d_grid_matches_pytorch():
    """
    WarpTracer 与 PyTorch 在同一 3D grid GRIN 场中应数值一致。
    """
    import numpy as np

    device = torch.device("cuda")

    Nx, Ny, Nz = 16, 8, 8
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
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.3, 0.3, -0.3, 0.3),
        polarization="s", device=device)

    tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out_pt = tp.trace(rays, max_steps=200, ds_init=0.02,
                       ds_min=0.02, ds_max=0.02, rk_tol=1e9,
                       fresnel=False)

    tw = WarpTracer(verts, faces, fm, materials, device)
    out_wp = tw.trace(rays, max_steps=200, ds_init=0.02)

    S_min = min(out_pt["positions"].shape[1], out_wp["positions"].shape[1])
    diff = (out_pt["positions"][:, :S_min] -
            out_wp["positions"][:, :S_min]).abs().max().item()

    print("\n[Test 2] Warp vs PyTorch GRIN 3D grid")
    print("  PyTorch steps: {}".format(out_pt["positions"].shape[1]))
    print("  Warp steps   : {}".format(out_wp["positions"].shape[1]))
    print("  max |Δ position|: {:.3e}".format(diff))

    assert diff < 1e-2, \
        "Warp vs PyTorch GRIN 3D mismatch: {:.6e}".format(diff)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))