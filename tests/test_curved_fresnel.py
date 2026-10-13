"""
tests/test_curved_fresnel.py
=============================
曲面界面上的 Fresnel 反射率验证。
"""
import math
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_sphere(device, radius=1.0, subdivisions=5):
    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    verts = torch.tensor(s.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(s.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def fresnel_rs_rp(n1, n2, theta_i):
    sin_t = n1 * math.sin(theta_i) / n2
    if sin_t >= 1.0:
        return 1.0, 1.0
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return r_s ** 2, r_p ** 2


def test_spherical_fresnel_s_polarization():
    """
    球面上的 Fresnel 反射率应与局部入射角匹配的解析值。
    使用 s 偏振光线，起点明确使入射角为 30°。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_sphere(device, radius=1.0, subdivisions=5)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 起点 (-3, 0.5, 0)，方向 +x，命中球面
    # 命中点 x² + 0.25 = 1 → x = -√0.75 ≈ -0.866
    # 局部法线 = 命中点位置 / |命中点| = (-0.866, 0.5, 0)
    # 入射角 = arccos(|d·n|) = arccos(0.866) = 30°
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(-3.0, 0.5, 0.0),
        direction=(1.0, 0.0, 0.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    R_measured = out["reflectivity"][0, 1, 0].item()

    # 用局部入射角 30° 计算解析 R_s
    theta_local = math.radians(30.0)
    R_s, _ = fresnel_rs_rp(1.0, 1.5, theta_local)

    print("\n[Test 1] Spherical Fresnel (s-polarization)")
    print("  Local θi ≈ 30°")
    print("  R_s analytic = {:.6f}".format(R_s))
    print("  R measured   = {:.6f}".format(R_measured))

    # 由于球面是离散三角形，局部法线有误差，放宽到 5%
    assert abs(R_measured - R_s) < 0.05, \
        "Spherical Fresnel mismatch: {:.6f} vs {:.6f}".format(R_measured, R_s)


def test_spherical_fresnel_normal_incidence():
    """法向入射（穿透球心方向）R 应匹配解析值。"""
    device = torch.device("cpu")
    verts, faces, fm = make_sphere(device, radius=1.0, subdivisions=5)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 从 x=-3 沿 +x 命中 x=-1 点（球面正前方）
    # 局部法线 = (-1, 0, 0)，入射角 = 0°
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(-3.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    R_measured = out["reflectivity"][0, 1, 0].item()
    R_expected = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2  # = 0.04

    print("\n[Test 2] Spherical Fresnel (normal incidence)")
    print("  R_expected = {:.6f}".format(R_expected))
    print("  R_measured = {:.6f}".format(R_measured))

    assert abs(R_measured - R_expected) < 0.01


def test_spherical_polarization_varies_with_incidence():
    """不同入射角的偏振响应不同（高 NA 效应）。"""
    device = torch.device("cpu")
    verts, faces, fm = make_sphere(device, radius=1.0, subdivisions=5)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    # 近轴（y=0.1）
    rays_n = generate_rays(
        source_type="grid", num_rays=1,
        origin=(-3.0, 0.1, 0.0), direction=(1.0, 0.0, 0.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)
    out_n = tracer.trace(rays_n, max_steps=3, surface_model="fresnel")

    # 边缘（y=0.9，接近球面边缘）
    rays_m = generate_rays(
        source_type="grid", num_rays=1,
        origin=(-3.0, 0.9, 0.0), direction=(1.0, 0.0, 0.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)
    out_m = tracer.trace(rays_m, max_steps=3, surface_model="fresnel")

    R_n = out_n["reflectivity"][0, 1, 0].item()
    R_m = out_m["reflectivity"][0, 1, 0].item()

    print("\n[Test 3] Spherical polarization varies with incidence")
    print("  Paraxial R: {:.6f}".format(R_n))
    print("  Marginal R: {:.6f}".format(R_m))

    # 边缘光线的入射角更大，R 更高
    assert R_m > R_n, \
        "Marginal ray should have higher reflectance: {:.4f} vs {:.4f}".format(
            R_m, R_n)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))