"""
tests/test_curved_geometry.py
==============================
球面/曲面界面上的 Fresnel、Snell 与聚焦验证。
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_sphere(device, radius=1.0, subdivisions=4):
    """高分辨率球面，用于曲面界面的光线交互。"""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    verts = torch.tensor(sphere.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(sphere.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0   # 外部 = air
    fm[:, 1] = 1   # 内部 = glass
    return verts, faces, fm


def make_spherical_lens(device, R=1.0, half_height=0.9):
    """双凸球面透镜的简化模型：两个球冠拼接。"""
    # 用两个 icosphere 的截面构造
    left = trimesh.creation.icosphere(subdivisions=3, radius=R)
    right = trimesh.creation.icosphere(subdivisions=3, radius=R)
    # 保留每个球的前半球
    left_v = left.vertices
    left_f = left.faces
    keep_l = left_v[left_f].mean(axis=1)[:, 2] < 0
    left = trimesh.Trimesh(vertices=left_v, faces=left_f[keep_l])

    right_v = right.vertices
    right_f = right.faces
    keep_r = right_v[right_f].mean(axis=1)[:, 2] > 0
    right = trimesh.Trimesh(vertices=right_v, faces=right_f[keep_r])
    right.apply_translation([0.0, 0.0, 1.0])

    combined = trimesh.util.concatenate([left, right])
    verts = torch.tensor(combined.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


# ==============================================================================
# Test 1: 球面界面上的 Snell 定律
# ==============================================================================

def test_spherical_interface_snell():
    """
    光线以 30° 入射角命中球面前表面，折射应满足局部 Snell 定律。
    球面的局部法线在命中点处。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_sphere(device, radius=1.0, subdivisions=5)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(30.0)
    # 光线从外侧朝球心打，方向 +x 分量 sin(30)
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(-2.0, 0.0, 0.0),
        direction=(math.cos(theta_i), math.sin(theta_i), 0.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="custom")

    pos = out["positions"][0]
    d_before = pos[1] - pos[0]
    d_before = d_before / d_before.norm()

    # 命中点 = pos[1]
    hit_pt = pos[1]
    local_normal = hit_pt / hit_pt.norm()

    d_after = pos[2] - pos[1]
    d_after = d_after / d_after.norm()

    cos_i = abs((d_before * local_normal).sum().item())
    cos_t = abs((d_after * local_normal).sum().item())

    sin_i = math.sqrt(max(0.0, 1.0 - cos_i ** 2))
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t ** 2))

    lhs = 1.0 * sin_i
    rhs = 1.5 * sin_t

    print("\n[Test 1] Spherical interface Snell")
    print("  theta_i (local) = {:.3f}°".format(math.degrees(math.acos(cos_i))))
    print("  n1·sinθi = {:.6f}, n2·sinθt = {:.6f}".format(lhs, rhs))

    assert abs(lhs - rhs) < 1e-3, \
        "Spherical Snell violated: {:.6f} vs {:.6f}".format(lhs, rhs)


# ==============================================================================
# Test 2: 球面透镜聚焦
# ==============================================================================

def test_spherical_lens_focus():
    """
    球面双凸透镜应将平行光聚焦到焦点附近。

    简化几何：两个半径 R=1 的球面拼接。透镜的近似焦距
        f ≈ R / (2·(n-1)) = 1 / (2 × 0.5) = 1.0
    对于双凸透镜，f ≈ 1 / ((n-1)·(1/R1 + 1/R2)) = 1 / (0.5 × 2) = 1.0
    """
    device = torch.device("cpu")
    verts, faces, fm = make_spherical_lens(device, R=1.0)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 平行光沿 +z 射向透镜
    rays = generate_rays(
        source_type="grid", num_rays=64,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.4, 0.4, -0.4, 0.4),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=10, surface_model="custom")

    # 检查最终光线是否汇聚（最终位置相对于初始径向距离更小）
    initial_r = torch.sqrt(out["positions"][:, 0, 0]**2 + out["positions"][:, 0, 1]**2)
    final_r = torch.sqrt(out["positions"][:, -1, 0]**2 + out["positions"][:, -1, 1]**2)

    # 允许一部分光线未聚焦（边缘光线球差）
    focused_fraction = (final_r < initial_r).float().mean().item()

    print("\n[Test 2] Spherical lens focus")
    print("  Initial radial mean: {:.4f}".format(initial_r.mean().item()))
    print("  Final radial mean  : {:.4f}".format(final_r.mean().item()))
    print("  Focused fraction   : {:.2f}".format(focused_fraction))

    assert focused_fraction > 0.5, \
        "At least half of rays should be focused, got {:.2f}".format(focused_fraction)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))