"""
tests/test_io_completeness.py
==============================
输出完整性、I/O 精确性、STL 水密性。
"""
import os
import tempfile
import pytest
import torch
import trimesh
import numpy as np

from georay import (AdvancedRayTracerPBRTGPU, generate_rays,
                     save_trace_output_to_npz, export_ray_history_to_stl)


def make_cube(device):
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


# ==============================================================================
# Test 1: 输出键完整性
# ==============================================================================

def test_trace_output_keys_complete():
    """trace() 返回的所有文档化键都应存在且形状正确。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=10)

    expected_keys = [
        "positions", "stokes", "active", "wavelengths",
        "reflectivity", "transmissivity", "absorptivity",
        "n1", "n2", "polarization_deg", "dop",
        "energy", "rk_error", "ds_history", "energy_conserved",
    ]

    N = 16
    S = 11

    for key in expected_keys:
        assert key in out, "Missing key: {}".format(key)

    # Shape 检查
    assert out["positions"].shape == (N, S, 3)
    assert out["stokes"].shape == (N, S, 4)
    assert out["active"].shape == (N, S)
    assert out["energy"].shape == (N, S)
    assert out["reflectivity"].shape == (N, S, 1)
    assert out["n1"].shape == (N, S, 1)
    assert out["dop"].shape == (N, S, 1)

    print("\n[Test 1] Output keys complete: all {} keys present".format(
        len(expected_keys)))


# ==============================================================================
# Test 2: NPZ 往返精度
# ==============================================================================

def test_npz_roundtrip_exact():
    """保存 → 加载后，所有张量应 bit-identical。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=8,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=10)

    tensor_keys = [k for k, v in out.items() if isinstance(v, torch.Tensor)]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "trace.npz")
        save_trace_output_to_npz(out, path)

        with np.load(path) as loaded:
            for key in tensor_keys:
                saved = loaded[key]
                original = out[key].cpu().numpy()
                assert saved.shape == original.shape, \
                    "Shape mismatch for {}".format(key)
                assert np.array_equal(saved, original), \
                    "Value mismatch for {}".format(key)

    print("\n[Test 2] NPZ roundtrip: all {} tensors bit-identical".format(
        len(tensor_keys)))


# ==============================================================================
# Test 3: STL 导出的几何有效性
# ==============================================================================

def test_stl_export_valid_geometry():
    """导出的 STL 应有非零顶点和面。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=4,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "rays.stl")
        export_ray_history_to_stl(out, path, radius=0.02, sides=6)
        assert os.path.exists(path)

        loaded = trimesh.load(path)
        assert loaded.vertices.shape[0] > 0
        assert loaded.faces.shape[0] > 0

    print("\n[Test 3] STL export valid")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))