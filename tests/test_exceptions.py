"""测试自定义异常和输入验证。"""
import pytest
import torch

from georay import (
    AdvancedRayTracerPBRTGPU, WarpTracer, generate_rays,
    GeoRayError, ConfigurationError, InvalidMaterialError,
    InvalidRayError,
)


def test_invalid_material_negative_n():
    """n < 0 应抛 InvalidMaterialError。"""
    import trimesh
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32)
    faces = torch.tensor(cube.faces, dtype=torch.long)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long)

    with pytest.raises(InvalidMaterialError):
        AdvancedRayTracerPBRTGPU(verts, faces, fm, {0: {"n": -1.5}}, "cpu")


def test_invalid_material_is_value_error():
    """InvalidMaterialError 应同时是 ValueError（向后兼容）。"""
    assert issubclass(InvalidMaterialError, ValueError)
    assert issubclass(InvalidMaterialError, GeoRayError)


def test_invalid_ray_missing_keys():
    """ray_dict 缺字段应抛 InvalidRayError。"""
    import trimesh
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32)
    faces = torch.tensor(cube.faces, dtype=torch.long)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long)
    materials = {0: {"n": 1.0}}

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, "cpu")
    bad_rays = {"origins": torch.zeros((4, 3))}
    with pytest.raises(InvalidRayError, match="missing required keys"):
        tracer.trace(bad_rays)


def test_generate_rays_invalid_source_type():
    """非法 source_type 应抛 InvalidRayError。"""
    with pytest.raises(InvalidRayError, match="Invalid source_type"):
        generate_rays(source_type="invalid_type", num_rays=4)


def test_generate_rays_zero_direction():
    """零方向向量应抛 InvalidRayError。"""
    with pytest.raises(InvalidRayError, match="non-zero"):
        generate_rays(direction=(0.0, 0.0, 0.0), num_rays=4)


def test_generate_rays_medium_id_list():
    """medium_id 列表长度不匹配应抛 InvalidRayError。"""
    with pytest.raises(InvalidRayError, match="length"):
        generate_rays(num_rays=4, medium_id=[0, 1])   # 长度 2 != 4


def test_medium_id_from_inside():
    """从介质内部发射光线应正常工作。"""
    import trimesh
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32)
    faces = torch.tensor(cube.faces, dtype=torch.long)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long)
    fm[:, 0] = 0
    fm[:, 1] = 1
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        medium_id=1)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, "cpu")
    out = tracer.trace(rays, max_steps=5)

    # 第一次界面：n1 = 1.5（玻璃），n2 = 1.0（空气）
    assert abs(out["n1"][0, 1, 0].item() - 1.5) < 1e-4