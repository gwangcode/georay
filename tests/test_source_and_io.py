"""
tests/test_source_and_io.py
============================
Comprehensive testing for:
    - generate_rays for all source types (random, point, area, mesh)
    - Polarization angle input formats
    - I/O round-trips (NPZ, STL)

Run:
    pytest tests/test_source_and_io.py -v
"""
import math
import os
import tempfile

import numpy as np
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays
from georay.io import save_trace_output_to_npz, export_ray_history_to_stl


# ==============================================================================
# Helpers
# ==============================================================================

def make_flat_interface(device, half_size=10.0):
    hs = half_size
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


# ==============================================================================
# 1. Random source — disk sampling
# ==============================================================================

def test_source_random_disk():
    """Random source fills a disk of given radius perpendicular to main_dir."""
    device = torch.device("cpu")
    radius = 2.0
    center = (1.0, 1.0, 0.0)
    rays = generate_rays(
        source_type="random", num_rays=500, radius=radius,
        origin=center, direction=(0.0, 0.0, 1.0),
        device=device)

    origins = rays["origins"]
    c = torch.tensor(center, device=device)
    centered = origins - c
    dists = centered.norm(dim=-1)

    assert torch.all(dists <= radius + 1e-5), \
        f"sample outside disk: max dist = {dists.max().item()}"
    assert dists.max().item() > 0.9 * radius, "samples fail to reach the edge"
    assert torch.allclose(origins[:, 2], torch.zeros_like(origins[:, 2]), atol=1e-5)

    main = torch.tensor([0.0, 0.0, 1.0], device=device)
    assert torch.allclose(rays["dirs"], main.expand_as(rays["dirs"]), atol=1e-5)


# ==============================================================================
# 2. Point source — cone sampling
# ==============================================================================

def test_source_point_cone():
    """Point source emits from a single point into a cone of half-angle θ."""
    device = torch.device("cpu")
    angle_deg = 30.0
    rays = generate_rays(
        source_type="point_source", num_rays=500, angle_deg=angle_deg,
        origin=(0.5, -0.5, 0.0), direction=(0.0, 0.0, 1.0),
        device=device)

    assert torch.allclose(rays["origins"],
                          torch.tensor([0.5, -0.5, 0.0], device=device).expand_as(rays["origins"]),
                          atol=1e-6)

    main = torch.tensor([0.0, 0.0, 1.0], device=device)
    cos_dot = (rays["dirs"] * main).sum(dim=-1)
    cos_max = math.cos(math.radians(angle_deg))
    assert torch.all(cos_dot >= cos_max - 1e-5), \
        f"direction outside cone: min cos = {cos_dot.min().item()}, expected ≥ {cos_max}"
    assert cos_dot.min().item() < cos_max + 0.01, "cone is empty"


# ==============================================================================
# 3. Area source — rectangular sampling
# ==============================================================================

def test_source_area_bounds():
    """Area source distributes origins uniformly within a rectangle."""
    device = torch.device("cpu")
    rays = generate_rays(
        source_type="area_source", num_rays=500,
        bounds=(-1.0, 1.0, -2.0, 2.0),
        origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        device=device)

    o = rays["origins"]
    x = o[:, 0]
    y = o[:, 1]
    z = o[:, 2]

    assert torch.all((x >= -1.0 - 1e-5) & (x <= 1.0 + 1e-5))
    assert torch.all((y >= -2.0 - 1e-5) & (y <= 2.0 + 1e-5))
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-5)

    assert x.min().item() < -0.9 and x.max().item() > 0.9
    assert y.min().item() < -1.9 and y.max().item() > 1.9

    assert torch.all(rays["dirs"][:, 2] >= -1e-6)


# ==============================================================================
# 4. Mesh source — surface emission
# ==============================================================================

def test_source_mesh_surface():
    """Mesh source places origins on the mesh surface."""
    device = torch.device("cpu")
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    rays = generate_rays(
        source_type="mesh_source", num_rays=500, mesh=mesh,
        device=device)

    o = rays["origins"]
    max_abs = o.abs().max(dim=-1).values
    assert torch.allclose(max_abs, torch.full_like(max_abs, 0.5), atol=1e-5), \
        f"origin not on unit box surface: max_abs range " \
        f"[{max_abs.min().item()}, {max_abs.max().item()}]"

    norms = rays["dirs"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ==============================================================================
# 5. Polarization input formats
# ==============================================================================

def test_polarization_angle_formats():
    """Verify Stokes vector assignments for various polarization specifications."""
    device = torch.device("cpu")

    def stokes_of(pol):
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization=pol, device=device)
        return rays["stokes"][0]

    s = stokes_of(0.0)
    assert abs(s[0].item() - 1.0) < 1e-6
    assert abs(s[1].item() - 1.0) < 1e-6
    assert abs(s[2].item()) < 1e-6

    s = stokes_of(45.0)
    assert abs(s[1].item()) < 1e-6
    assert abs(s[2].item() - 1.0) < 1e-6

    s = stokes_of(90.0)
    assert abs(s[1].item() + 1.0) < 1e-6
    assert abs(s[2].item()) < 1e-6

    s = stokes_of(-45.0)
    assert abs(s[1].item()) < 1e-6
    assert abs(s[2].item() + 1.0) < 1e-6

    s = stokes_of("s")
    assert abs(s[1].item() - 1.0) < 1e-6

    s = stokes_of("p")
    assert abs(s[1].item() + 1.0) < 1e-6

    s = stokes_of("random")
    assert abs(s[1].item()) < 1e-6
    assert abs(s[2].item()) < 1e-6


# ==============================================================================
# 6. NPZ save/load round-trip
# ==============================================================================

def test_npz_save_load_roundtrip():
    """
    save_trace_output_to_npz writes a valid NPZ with all tensor keys.

    ★ 修正：使用 with 语句显式关闭 np.load 返回的 NpzFile，
       否则 Windows 上文件句柄不释放，TemporaryDirectory 清理时会
       报 WinError 32 (PermissionError)。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }
    rays = generate_rays(
        source_type="grid", num_rays=4,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.01, 0.01, -0.01, 0.01),
        polarization="s", device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=4, fresnel=True)

    tensor_keys = [k for k, v in out.items() if isinstance(v, torch.Tensor)]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "trace.npz")
        save_trace_output_to_npz(out, path)
        assert os.path.exists(path), "NPZ file not created"

        # ★ with 语句确保 NpzFile 被关闭
        with np.load(path) as loaded:
            saved_keys = set(loaded.files)
            for k in tensor_keys:
                assert k in saved_keys, f"missing key in NPZ: {k}"

            assert np.allclose(loaded["positions"], out["positions"].cpu().numpy())
            assert np.allclose(loaded["energy"], out["energy"].cpu().numpy())
            assert np.allclose(loaded["stokes"], out["stokes"].cpu().numpy())


# ==============================================================================
# 7. STL export
# ==============================================================================

def test_stl_export():
    """export_ray_history_to_stl produces a valid STL file."""
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }
    rays = generate_rays(
        source_type="grid", num_rays=4,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.01, 0.01, -0.01, 0.01),
        polarization="s", device=device)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=4, fresnel=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ray_tracks.stl")
        export_ray_history_to_stl(out, path, radius=0.005, sides=4)
        assert os.path.exists(path), "STL file not created"

        # ★ with 确保 trimesh 句柄释放
        loaded = trimesh.load(path)
        assert loaded is not None
        assert loaded.vertices.shape[0] > 0
        assert loaded.faces.shape[0] > 0


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))