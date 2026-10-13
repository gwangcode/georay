"""
tests/test_ray_from_medium.py
==============================
光线从介质内部出发的测试。

当前 GeoRay 的 trace() 硬编码 curr_medium_ids = zeros(N)，
所有光线从空气出发。本文件测试"从介质内部出发"的能力，
需要 tracer 支持 ray_dict["medium_ids"] 字段。

如果 API 未实现，测试会 skip 并给出提示。
"""
import math
import inspect
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, WarpTracer, generate_rays


# ==============================================================================
# API 支持检测
# ==============================================================================

def _supports_medium_ids(cls):
    """检查 tracer 的 trace() 是否支持 ray_dict["medium_ids"]。"""
    try:
        src = inspect.getsource(cls.trace)
    except (OSError, TypeError):
        return False
    return "medium_ids" in src


_HAS_MEDIUM_API = _supports_medium_ids(AdvancedRayTracerPBRTGPU)

_skip_if_no_api = pytest.mark.skipif(
    not _HAS_MEDIUM_API,
    reason="tracer.trace() does not yet support ray_dict['medium_ids']"
)


# ==============================================================================
# 场景构造
# ==============================================================================

def make_glass_slab(device):
    """单层玻璃板 z ∈ [0, 1]，两侧是空气。"""
    verts = torch.tensor([
        [-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0],
        [-5.0, -5.0, 1.0], [5.0, -5.0, 1.0], [5.0, 5.0, 1.0], [-5.0, 5.0, 1.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)
    fm = torch.zeros((4, 2), dtype=torch.long, device=device)
    fm[0:2, 0] = 1   # z=0 平面：+z 侧是玻璃
    fm[0:2, 1] = 0   # -z 侧是空气
    fm[2:4, 0] = 0   # z=1 平面：+z 侧是空气
    fm[2:4, 1] = 1   # -z 侧是玻璃
    return verts, faces, fm


def make_glass_box(device):
    """玻璃立方体 [-1, 1]³。"""
    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


# ==============================================================================
# Test 1: 玻璃内部出发 → 从表面折射
# ==============================================================================

@_skip_if_no_api
def test_ray_from_glass_medium():
    """
    光线从玻璃板内部 (0.5) 出发，沿 +z 前进。
    应先命中 z=1 界面（玻璃→空气），按 Snell 定律折射。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_glass_slab(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    # ★ 声明初始介质 ID
    rays["medium_ids"] = torch.tensor([1], dtype=torch.long, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="custom")

    # 第一次交互应该是 z=1 界面，介质从玻璃 (n=1.5) 到空气 (n=1.0)
    n1 = out["n1"][0, 1, 0].item()
    n2 = out["n2"][0, 1, 0].item()

    print("\n[Test 1] Ray from glass medium")
    print("  n1={:.4f}, n2={:.4f}".format(n1, n2))

    assert abs(n1 - 1.5) < 1e-4, "n1 should be 1.5 (glass), got {}".format(n1)
    assert abs(n2 - 1.0) < 1e-4, "n2 should be 1.0 (air), got {}".format(n2)


# ==============================================================================
# Test 2: 从内部出发 → TIR
# ==============================================================================

@_skip_if_no_api
def test_ray_from_medium_tir():
    """
    光线从玻璃内部以大角度 (> θc ≈ 41.8°) 入射到界面。
    应发生 TIR，反射率 R ≈ 1。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_glass_box(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta = math.radians(50.0)   # > θc
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.0), direction=(math.sin(theta), 0.0, math.cos(theta)),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays["medium_ids"] = torch.tensor([1], dtype=torch.long, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="fresnel")

    R = out["reflectivity"][0, 1, 0].item()

    print("\n[Test 2] Ray from glass medium → TIR")
    print("  theta = 50°, R = {:.6f}".format(R))

    assert R > 0.99, "TIR expected, got R = {:.6f}".format(R)


# ==============================================================================
# Test 3: 从 GRIN 介质内部出发
# ==============================================================================

@_skip_if_no_api
def test_ray_from_grin_medium():
    """
    光线从 GRIN 介质内部出发，RK4 步进应从第一步就生效。
    与从外部入射的光线相比，内部出发的光线不应经过空气→GRIN 界面。
    """
    device = torch.device("cpu")
    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1

    def parabolic_grin(x, y, z, wl):
        return 1.5 - 0.1 * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": parabolic_grin},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays["medium_ids"] = torch.tensor([1], dtype=torch.long, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=50, ds_init=0.02, fresnel=False)

    # 第一次 step 应该是 GRIN 步进（无界面交互）
    hist_refl = out["reflectivity"][0, :, 0]
    hist_trans = out["transmissivity"][0, :, 0]

    print("\n[Test 3] Ray from GRIN medium")
    print("  step 1: refl={:.4f}, trans={:.4f}".format(
        hist_refl[1].item(), hist_trans[1].item()))

    # GRIN 内部步进不产生界面反射/透射
    assert hist_refl[1].item() < 1e-6, "GRIN step should have no reflection"
    assert hist_trans[1].item() < 1e-6, "GRIN step should have no transmission"


# ==============================================================================
# Test 4: 两条光线从不同介质同时出发
# ==============================================================================

@_skip_if_no_api
def test_two_rays_from_different_media():
    """
    两个 batch 的光线：一条从空气出发，一条从玻璃出发。
    追踪结果应反映不同的初始介质。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_glass_slab(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 从空气出发 (z=-0.5)
    rays_air = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays_air["medium_ids"] = torch.tensor([0], dtype=torch.long, device=device)

    # 从玻璃出发 (z=0.5)
    rays_glass = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays_glass["medium_ids"] = torch.tensor([1], dtype=torch.long, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    out_air = tracer.trace(rays_air, max_steps=5, surface_model="custom")
    out_glass = tracer.trace(rays_glass, max_steps=5, surface_model="custom")

    # 从空气出发：第一次界面是 air → glass
    n1_air = out_air["n1"][0, 1, 0].item()
    n2_air = out_air["n2"][0, 1, 0].item()

    # 从玻璃出发：第一次界面是 glass → air
    n1_glass = out_glass["n1"][0, 1, 0].item()
    n2_glass = out_glass["n2"][0, 1, 0].item()

    print("\n[Test 4] Two rays from different media")
    print("  From air  : n1={:.4f} → n2={:.4f}".format(n1_air, n2_air))
    print("  From glass: n1={:.4f} → n2={:.4f}".format(n1_glass, n2_glass))

    assert abs(n1_air - 1.0) < 1e-4 and abs(n2_air - 1.5) < 1e-4, \
        "Air ray: n1 should be 1.0, n2 should be 1.5"
    assert abs(n1_glass - 1.5) < 1e-4 and abs(n2_glass - 1.0) < 1e-4, \
        "Glass ray: n1 should be 1.5, n2 should be 1.0"


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))