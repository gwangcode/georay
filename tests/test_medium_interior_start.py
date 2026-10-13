"""
tests/test_medium_interior_start.py
====================================
光线从介质内部出发的场景测试。

覆盖：
    1. 光线从玻璃内部出发，正确识别 n1=1.5, n2=1.0
    2. 光线从 GRIN 介质内部出发，RK4 从第一步开始
    3. 光线在玻璃内部发生 TIR
    4. 光线从玻璃内部穿出到空气

Run:
    pytest tests/test_medium_interior_start.py -v -s
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


_HAS_CUDA = torch.cuda.is_available()

try:
    from georay import WarpTracer
    _HAS_WARP_TRACER = WarpTracer is not None
except ImportError:
    _HAS_WARP_TRACER = False


# ==============================================================================
# 场景构造
# ==============================================================================

def make_glass_box(device, extent=2.0):
    """玻璃盒，外部空气，内部玻璃。"""
    box = trimesh.creation.box(extents=[extent, extent, extent])
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0   # 外部
    fm[:, 1] = 1   # 内部玻璃
    return verts, faces, fm


# ==============================================================================
# Test 1: 光线从玻璃内部出发
# ==============================================================================

def test_ray_starts_inside_glass():
    """
    光线从玻璃盒内部出发，方向 +z。
    第一次命中玻璃-空气界面时：
      - n1 = 1.5（玻璃）
      - n2 = 1.0（空气）
      - 若入射角 > θc，应发生 TIR
    """
    device = torch.device("cpu")
    extent = 2.0
    verts, faces, fm = make_glass_box(device, extent)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 光线从玻璃盒内部（原点）出发，方向 +z
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device,
        medium_id=1)   # ★ 从玻璃内部出发

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="fresnel")

    # 第一次界面：n1 = 1.5（玻璃），n2 = 1.0（空气）
    n1_step1 = out["n1"][0, 1, 0].item()
    n2_step1 = out["n2"][0, 1, 0].item()

    print("\n[Test 1] Ray starts inside glass")
    print("  n1 at first interface: {:.4f}".format(n1_step1))
    print("  n2 at first interface: {:.4f}".format(n2_step1))

    assert abs(n1_step1 - 1.5) < 0.01, \
        "n1 should be 1.5 (glass), got {:.4f}".format(n1_step1)
    assert abs(n2_step1 - 1.0) < 0.01, \
        "n2 should be 1.0 (air), got {:.4f}".format(n2_step1)

    assert out["energy_conserved"]


# ==============================================================================
# Test 2: 光线从玻璃内部发生 TIR
# ==============================================================================

def test_ray_inside_glass_tir():
    """
    光线从玻璃内部（n=1.5）出发，以 50° 入射角（> θc ≈ 41.8°）射向界面。
    应该发生 TIR，反射率 R ≈ 1。
    """
    device = torch.device("cpu")
    extent = 4.0
    verts, faces, fm = make_glass_box(device, extent)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta = math.radians(50.0)
    direction = (math.sin(theta), 0.0, math.cos(theta))

    # 从玻璃内部出发
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.0), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device,
        medium_id=1)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    R = out["reflectivity"][0, 1, 0].item()

    print("\n[Test 2] TIR from inside glass")
    print("  theta_i = 50 deg, theta_c ≈ 41.8 deg")
    print("  R measured = {:.6f}".format(R))

    assert R > 0.99, "TIR should occur: R = {:.4f}".format(R)


# ==============================================================================
# Test 3: 光线从玻璃内部穿出到空气
# ==============================================================================

def test_ray_inside_glass_escapes():
    """法向入射下，光线从玻璃内部穿出到空气（无 TIR）。"""
    device = torch.device("cpu")
    extent = 2.0
    verts, faces, fm = make_glass_box(device, extent)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 法向入射，从玻璃内部出发
    rays = generate_rays(
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device,
        medium_id=1)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=10, surface_model="fresnel")

    # 所有光线都应逃逸
    final_active = out["active"][:, -1]
    n_stuck = final_active.sum().item()

    print("\n[Test 3] Ray from inside glass escapes")
    print("  Escaped: {}/16".format(16 - n_stuck))

    assert n_stuck == 0, "All rays should escape"
    assert out["energy_conserved"]


# ==============================================================================
# Test 4: 光线从 GRIN 介质内部出发
# ==============================================================================

def test_ray_inside_grin_starts_stepping():
    """
    光线从抛物线 GRIN 介质内部出发，离轴起点。
    RK4 步进应从第一步就对光线施加弯折。
    """
    device = torch.device("cpu")
    box = trimesh.creation.box(extents=[2.0, 2.0, 4.0])
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1

    def parabolic_grin(x, y, z, wavelength):
        return 1.5 - 0.15 * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {
            "n": 1.5,
            "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
            "grin_field": parabolic_grin,
        },
    }

    # 从 GRIN 介质内部（x=0.3, y=0, z=-1.0）出发
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.3, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device,
        medium_id=1)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=200, ds_init=0.02,
                        ds_min=1e-4, ds_max=0.05,
                        fresnel=False)

    pos = out["positions"][0]
    x_start = pos[0, 0].item()
    x_end = pos[-1, 0].item()

    print("\n[Test 4] Ray starts inside GRIN medium")
    print("  x_start = {:.4f}, x_end = {:.4f}".format(x_start, x_end))

    # 从内侧出发，x 应向轴收敛
    assert abs(x_end) < abs(x_start), \
        "GRIN should bend ray: x_start={:.4f}, x_end={:.4f}".format(
            x_start, x_end)


# ==============================================================================
# Test 5: WarpTracer 从介质内部出发
# ==============================================================================

@pytest.mark.skipif(not (_HAS_CUDA and _HAS_WARP_TRACER),
                     reason="WarpTracer requires CUDA + warp-lang")
def test_warp_ray_inside_glass():
    """WarpTracer 也应支持从介质内部出发。"""
    device = torch.device("cuda")
    extent = 2.0
    verts, faces, fm = make_glass_box(device, extent)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device,
        medium_id=1)

    tracer = WarpTracer(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=10, surface_model="fresnel")

    n1_step1 = out["n1"][0, 1, 0].item()
    n2_step1 = out["n2"][0, 1, 0].item()

    print("\n[Test 5] WarpTracer from inside glass")
    print("  n1 = {:.4f}, n2 = {:.4f}".format(n1_step1, n2_step1))

    assert abs(n1_step1 - 1.5) < 0.01
    assert abs(n2_step1 - 1.0) < 0.01


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))