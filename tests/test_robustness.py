"""
tests/test_robustness.py
=========================
数值鲁棒性与边界条件测试。
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


# ==============================================================================
# Helpers
# ==============================================================================

def make_cube(device):
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


DEFAULT_MATERIALS = {
    0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
}


# ==============================================================================
# Test 1: 零长度方向向量
# ==============================================================================

def test_zero_direction_vector():
    """
    零长度方向向量不应导致 NaN，tracer 应优雅处理。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    # 手动构造 rays，方向是 (0, 0, 0)
    N = 3
    rays = {
        "origins": torch.tensor([[0.0, 0.0, -3.0]] * N,
                                 dtype=torch.float32, device=device),
        "dirs": torch.zeros(N, 3, dtype=torch.float32, device=device),
        "wavelengths": torch.full((N, 1), 550.0,
                                    dtype=torch.float32, device=device),
        "stokes": torch.tensor([[1.0, 1.0, 0.0, 0.0]] * N,
                                dtype=torch.float32, device=device),
        "s_ref": torch.tensor([[0.0, 1.0, 0.0]] * N,
                               dtype=torch.float32, device=device),
        "p_ref": torch.tensor([[1.0, 0.0, 0.0]] * N,
                               dtype=torch.float32, device=device),
        "ray_ids": torch.arange(N, dtype=torch.long, device=device),
    }

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, DEFAULT_MATERIALS, device)

    # 应该要么优雅返回，要么抛出明确的异常
    try:
        out = tracer.trace(rays, max_steps=5)
        # 检查输出无 NaN
        assert not torch.isnan(out["positions"]).any(), "NaN in positions"
        print("\n[Test 1] Zero direction: gracefully handled")
    except (ValueError, RuntimeError) as e:
        print("\n[Test 1] Zero direction: raised {}".format(type(e).__name__))
        # 报错也可接受


# ==============================================================================
# Test 2: NaN 输入
# ==============================================================================

def test_nan_input_detection():
    """
    NaN 位置输入应被检测或产生可预测行为。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    N = 2
    rays = {
        "origins": torch.tensor([[float("nan"), 0.0, -3.0],
                                  [0.0, 0.0, -3.0]],
                                 dtype=torch.float32, device=device),
        "dirs": torch.tensor([[0.0, 0.0, 1.0]] * N,
                              dtype=torch.float32, device=device),
        "wavelengths": torch.full((N, 1), 550.0,
                                    dtype=torch.float32, device=device),
        "stokes": torch.tensor([[1.0, 1.0, 0.0, 0.0]] * N,
                                dtype=torch.float32, device=device),
        "s_ref": torch.tensor([[0.0, 1.0, 0.0]] * N,
                               dtype=torch.float32, device=device),
        "p_ref": torch.tensor([[1.0, 0.0, 0.0]] * N,
                               dtype=torch.float32, device=device),
        "ray_ids": torch.arange(N, dtype=torch.long, device=device),
    }

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, DEFAULT_MATERIALS, device)

    try:
        out = tracer.trace(rays, max_steps=5)
        # NaN 输入的那条光线应不参与命中（hit_face = -1）或位置不变
        print("\n[Test 2] NaN input: completed without crash")
        # 至少第二条光线应正常
        assert not torch.isnan(out["positions"][1]).any(), \
            "Valid ray produced NaN"
    except (ValueError, RuntimeError) as e:
        print("\n[Test 2] NaN input: raised {}".format(type(e).__name__))


# ==============================================================================
# Test 3: float64 输入
# ==============================================================================

def test_float64_input_conversion():
    """
    float64 输入应被正确处理（转换为 float32 或报错）。
    """
    device = torch.device("cpu")
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])

    # 用 float64 传入
    verts = torch.tensor(cube.vertices, dtype=torch.float64, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1

    # tracer 应该接受并内部转换为 float32
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, DEFAULT_MATERIALS, device)

    rays = generate_rays(
        source_type="grid", num_rays=4,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.1, 0.1, -0.1, 0.1),
        polarization="s", device=device)

    out = tracer.trace(rays, max_steps=5)

    print("\n[Test 3] float64 input: converted to float32")
    assert out["positions"].dtype == torch.float32
    assert not torch.isnan(out["positions"]).any()


# ==============================================================================
# Test 4: 空网格
# ==============================================================================

def test_empty_mesh():
    """
    0 面网格不应导致崩溃。
    """
    device = torch.device("cpu")
    verts = torch.zeros((0, 3), dtype=torch.float32, device=device)
    faces = torch.zeros((0, 3), dtype=torch.long, device=device)
    fm = torch.zeros((0, 2), dtype=torch.long, device=device)

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    try:
        tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        rays = generate_rays(
            source_type="grid", num_rays=4,
            origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
            bounds=(-0.1, 0.1, -0.1, 0.1),
            polarization="s", device=device)
        out = tracer.trace(rays, max_steps=5)

        # 所有光线应直接逃逸（无命中）
        active = out["active"][:, -1]
        assert not active.any(), "Empty mesh: rays should all escape"

        print("\n[Test 4] Empty mesh: gracefully handled, "
              "all rays escaped")
    except Exception as e:
        print("\n[Test 4] Empty mesh raised {}: {}".format(type(e).__name__, e))
        pytest.fail("Empty mesh should not crash")


# ==============================================================================
# Test 5: 极端坐标
# ==============================================================================

def test_extreme_coordinates():
    """
    极大/极小坐标不应导致浮点溢出。
    """
    device = torch.device("cpu")
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, DEFAULT_MATERIALS, device)

    # 光线起点 1e6 远
    rays = generate_rays(
        source_type="grid", num_rays=2,
        origin=(0.0, 0.0, -1e6), direction=(0.0, 0.0, 1.0),
        bounds=(-0.1, 0.1, -0.1, 0.1),
        polarization="s", device=device)

    out = tracer.trace(rays, max_steps=5)

    print("\n[Test 5] Extreme coordinates (1e6 distance)")
    assert not torch.isnan(out["positions"]).any(), "NaN produced"
    assert not torch.isinf(out["positions"]).any(), "Inf produced"


# ==============================================================================
# Test 6: 零步数
# ==============================================================================

def test_zero_max_steps():
    """max_steps=0 应返回初始状态。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, DEFAULT_MATERIALS, device)
    rays = generate_rays(
        source_type="grid", num_rays=4,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.1, 0.1, -0.1, 0.1),
        polarization="s", device=device)

    out = tracer.trace(rays, max_steps=0)

    print("\n[Test 6] max_steps=0")
    assert out["positions"].shape[1] == 1
    assert torch.all(out["active"][:, 0]), "All rays should be initially active"
    assert torch.allclose(out["energy"][:, 0],
                          torch.ones_like(out["energy"][:, 0]))


# ==============================================================================
# Test 7: 重复调用同一 tracer
# ==============================================================================

def test_repeated_trace_calls():
    """同一 tracer 多次调用 trace() 应产生一致结果。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, DEFAULT_MATERIALS, device)
    rays = generate_rays(
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    # 用相同 seed 跑两次
    torch.manual_seed(42)
    out1 = tracer.trace(rays, max_steps=10, surface_model="fresnel")

    torch.manual_seed(42)
    out2 = tracer.trace(rays, max_steps=10, surface_model="fresnel")

    print("\n[Test 7] Repeated trace calls")
    diff = (out1["positions"] - out2["positions"]).abs().max().item()
    print("  max |Delta| between two calls: {:.3e}".format(diff))

    assert diff < 1e-5, "Same seed should produce identical results"


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))