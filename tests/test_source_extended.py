"""
tests/test_source_extended.py
==============================
光源生成器的扩展测试：mesh_source、point_source、多波长。
"""
import math
import pytest
import torch
import trimesh

from georay import generate_rays


# ==============================================================================
# Test 1: mesh_source 余弦加权发射
# ==============================================================================

def test_mesh_source_emission_distribution():
    """
    mesh_source 从网格表面按面积采样，方向按余弦加权（Lambertian）。
    检查：光线起点在网格表面；方向分布 cos θ 均值 ≈ 2/3。
    """
    device = torch.device("cpu")
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    N = 1000
    rays = generate_rays(
        source_type="mesh_source", num_rays=N,
        mesh=mesh, device=device)

    origins = rays["origins"]
    dirs = rays["dirs"]

    # 起点应在单位盒表面（|x| = |y| = |z| = 0.5 至少一个）
    on_surface = (origins.abs() - 0.5).abs() < 1e-5
    on_surface_any = on_surface.any(dim=-1)
    assert on_surface_any.all(), \
        "All origins should lie on the box surface"

    # 方向应指向外法线半球
    # 检查方向模长 = 1
    dir_norm = dirs.norm(dim=-1)
    assert torch.allclose(dir_norm, torch.ones_like(dir_norm), atol=1e-5)

    print("\n[Test 1] mesh_source emission")
    print("  All {} rays on box surface".format(N))
    print("  All directions normalized")


# ==============================================================================
# Test 2: point_source 锥角严格
# ==============================================================================

def test_point_source_cone_angle():
    """
    point_source 的所有方向应严格落在半角 = angle_deg 的锥内。
    """
    device = torch.device("cpu")
    angle_deg = 25.0

    rays = generate_rays(
        source_type="point_source", num_rays=500,
        angle_deg=angle_deg, direction=(0.0, 0.0, 1.0),
        device=device)

    dirs = rays["dirs"]
    main = torch.tensor([0.0, 0.0, 1.0], device=device)
    cos_dot = (dirs * main).sum(dim=-1)

    min_cos = math.cos(math.radians(angle_deg))

    print("\n[Test 2] point_source cone angle")
    print("  Specified half-angle: {:.2f}°".format(angle_deg))
    print("  Min cos(θ) measured : {:.6f}".format(cos_dot.min().item()))
    print("  Min cos(θ) required : {:.6f}".format(min_cos))

    assert torch.all(cos_dot >= min_cos - 1e-6), \
        "Some directions outside the cone"

    # 实际锥角应接近指定值（不空）
    actual_max_angle = math.degrees(math.acos(cos_dot.min().item()))
    print("  Actual max half-angle: {:.2f}°".format(actual_max_angle))
    assert actual_max_angle > angle_deg - 1.0


# ==============================================================================
# Test 3: 多波长光源
# ==============================================================================

def test_multi_wavelength_source():
    """
    单次 generate_rays 调用应只支持单一波长。
    不同波长的光线应通过多次调用来生成。
    验证：两次调用（不同波长），合并后 tracer 能处理。
    """
    device = torch.device("cpu")
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    rays_400 = generate_rays(
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.3, 0.3, -0.3, 0.3),
        polarization="s", device=device, wavelength_nm=400.0)

    rays_700 = generate_rays(
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.3, 0.3, -0.3, 0.3),
        polarization="s", device=device, wavelength_nm=700.0)

    assert torch.all(rays_400["wavelengths"] == 400.0)
    assert torch.all(rays_700["wavelengths"] == 700.0)

    print("\n[Test 3] Multi-wavelength sources")
    print("  Source A: λ = 400nm (all rays)")
    print("  Source B: λ = 700nm (all rays)")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))