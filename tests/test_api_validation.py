"""
tests/test_api_validation.py
==============================
API 配置验证：非法配置应在构造时立即报错。
"""
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU


def make_cube(device):
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


# ==============================================================================
# Test 1: 空 materials_cfg
# ==============================================================================

def test_empty_materials_cfg():
    """空 materials_cfg 应立即报错。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    with pytest.raises(ValueError, match="non-empty"):
        AdvancedRayTracerPBRTGPU(verts, faces, fm, {}, device)

    print("\n[Test 1] Empty materials_cfg: correctly rejected")


# ==============================================================================
# Test 2: face_materials 引用未定义的材质 ID
# ==============================================================================

def test_face_materials_references_undefined_id():
    """face_materials 引用了 materials_cfg 中没有的 ID 应报错。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)
    fm[:, 1] = 5   # ★ 但 materials 只有 0 和 1

    materials = {
        0: {"n": 1.0},
        1: {"n": 1.5},
    }

    with pytest.raises(ValueError, match="undefined material IDs"):
        AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 2] Undefined material ID: correctly rejected")


# ==============================================================================
# Test 3: 负折射率
# ==============================================================================

def test_negative_refractive_index():
    """n < 0 应报错。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    materials = {
        0: {"n": 1.0},
        1: {"n": -1.5},   # ★ 非法
    }

    with pytest.raises(ValueError, match="must be > 0"):
        AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 3] Negative n: correctly rejected")


# ==============================================================================
# Test 4: reflectivity 超出 [0, 1]
# ==============================================================================

def test_reflectivity_out_of_range():
    """reflectivity > 1 应报错。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    materials = {
        0: {"n": 1.0},
        1: {"n": 1.5, "reflectivity": 1.5},   # ★ 非法
    }

    with pytest.raises(ValueError, match="reflectivity.*must be in"):
        AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 4] Out-of-range reflectivity: correctly rejected")


# ==============================================================================
# Test 5: absorptivity 为负数
# ==============================================================================

def test_negative_absorptivity():
    """absorptivity < 0 应报错。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    materials = {
        0: {"n": 1.0},
        1: {"n": 1.5, "absorptivity": -0.1},
    }

    with pytest.raises(ValueError, match="absorptivity.*must be in"):
        AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 5] Negative absorptivity: correctly rejected")


# ==============================================================================
# Test 6: grin_field 类型错误
# ==============================================================================

def test_grin_field_wrong_type():
    """grin_field 不是 dict 或 callable 应报错。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    materials = {
        0: {"n": 1.0},
        1: {"n": 1.5, "grin_field": 42},   # ★ 非法
    }

    with pytest.raises(ValueError, match="must be dict or callable"):
        AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 6] Invalid grin_field type: correctly rejected")


# ==============================================================================
# Test 7: grin_field dict 类型未知
# ==============================================================================

def test_grin_field_unknown_type():
    """grin_field['type'] 不在白名单应报错。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    materials = {
        0: {"n": 1.0},
        1: {"n": 1.5, "grin_field": {"type": "unknown_type"}},
    }

    with pytest.raises(ValueError, match="must be one of"):
        AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 7] Unknown grin_field type: correctly rejected")


# ==============================================================================
# Test 8: grin_field parabolic 负 A
# ==============================================================================

def test_grin_parabolic_negative_A():
    """parabolic GRIN 的 A < 0 应报错。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    materials = {
        0: {"n": 1.0},
        1: {"n": 1.5,
            "grin_field": {"type": "parabolic", "n0": 1.5, "A": -0.1}},
    }

    with pytest.raises(ValueError, match="A.*must be >= 0"):
        AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 8] Parabolic GRIN with A < 0: correctly rejected")


# ==============================================================================
# Test 9: 合法配置应正常构造
# ==============================================================================

def test_valid_config_constructs():
    """合法配置应正常构造 tracer。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    print("\n[Test 9] Valid config: constructed successfully")
    assert tracer is not None


# ==============================================================================
# Test 10: surface_model 非法值
# ==============================================================================

def test_invalid_surface_model():
    """surface_model 不是合法值时应在 trace() 报错。"""
    device = torch.device("cpu")
    verts, faces, fm = make_cube(device)

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    from georay import generate_rays
    rays = generate_rays(
        source_type="grid", num_rays=4,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    with pytest.raises(ValueError, match="surface_model"):
        tracer.trace(rays, max_steps=5, surface_model="invalid_mode")

    print("\n[Test 10] Invalid surface_model: correctly rejected")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))