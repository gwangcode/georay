"""
tests/test_branch_energy.py
============================
光线分裂能量守恒的等价验证。

当前 GeoRay 每条光线只走一个 Monte Carlo 分支。
但分支概率乘加应严格守恒：
    P_specular + P_diffuse + P_refract + P_absorb = 1
且对所有 N 条光线的平均：
    ⟨w_next/w_prev⟩ = ⟨branch_prob⟩ = 1（期望意义上）
"""
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_flat_interface(device, hs=100.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def test_branch_probabilities_sum_to_one():
    """
    每个界面的 P_absorb + P_diffuse + P_specular + P_refract = 1。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.8, "diffuse": 0.1, "absorptivity": 0.05},
        1: {"n": 1.5, "reflectivity": 0.8, "diffuse": 0.1, "absorptivity": 0.05},
    }

    rays = generate_rays(
        source_type="grid", num_rays=64,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    R = out["reflectivity"][:, 1, 0]
    T = out["transmissivity"][:, 1, 0]
    A = out["absorptivity"][:, 1, 0]
    total = R + T + A

    interacted = total > 1e-6
    if interacted.any():
        max_dev = (total[interacted] - 1.0).abs().max().item()

        print("\n[Test 1] Branch probabilities sum to 1")
        print("  Interface events: {}".format(interacted.sum().item()))
        print("  max |sum - 1|: {:.3e}".format(max_dev))

        assert max_dev < 1e-4


def test_branch_energy_expectation():
    """
    Monte Carlo 单路径追踪中，权重期望恒为 1。
    即 Σ E_final / N ≈ 1（在足够多光线下）。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    N = 4096
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    torch.manual_seed(0)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    final_mean = out["energy"][:, -1].mean().item()

    print("\n[Test 2] Branch energy expectation")
    print("  N = {}".format(N))
    print("  Mean final weight: {:.6f}".format(final_mean))

    # 期望权重 = 1（因为 R+T=1）
    # 但每条光线只走一个分支，所以 N 条光线的平均 ≈ 1
    assert abs(final_mean - 1.0) < 0.02


def test_branch_probability_ratio():
    """
    分支概率比值应匹配解析 Fresnel：
    P_specular / P_refract = R / (1-R)  （无吸收、无漫反射时）
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    import math
    theta_i = math.radians(45.0)

    rays = generate_rays(
        source_type="grid", num_rays=64,
        origin=(0.0, 0.0, -1.0),
        direction=(math.sin(theta_i), 0.0, math.cos(theta_i)),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    R = out["reflectivity"][:, 1, 0]
    T = out["transmissivity"][:, 1, 0]

    # R / (R+T) = P_specular / (P_specular + P_refract)
    ratio = (R / (R + T + 1e-12)).mean().item()

    # 解析：R_s(45°) / 1.0
    n1, n2 = 1.0, 1.5
    sin_t = n1 * math.sin(theta_i) / n2
    cos_t = math.sqrt(1 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    R_expected = r_s ** 2

    print("\n[Test 3] Branch probability ratio")
    print("  P_spec / (P_spec + P_refract) = {:.6f}".format(ratio))
    print("  Analytic R_s = {:.6f}".format(R_expected))

    assert abs(ratio - R_expected) < 0.01


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))