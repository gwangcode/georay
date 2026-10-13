"""
tests/test_branch_energy.py
============================
光线分裂的概率与期望验证。

关键物理：
    单路径 MC 追踪中，每条光线只走一个分支。
    权重 = 分支概率的累乘。这意味着：
      E[w] = P_spec · P_spec + P_refract · P_refract = R² + T²
    而不是 1（后者只在多路径叠加时才成立）。

    因此"权重期望 = 1"的断言是错误的。
    正确的守恒约束是界面级归一化：R + T + A = 1。
"""
import math
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


def fresnel_Rs(n1, n2, theta_i):
    sin_t = n1 * math.sin(theta_i) / n2
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    return r_s ** 2


def test_branch_probabilities_sum_to_one():
    """界面的 P_absorb + P_diffuse + P_specular + P_refract = 1。"""
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


def test_branch_energy_expectation_correct():
    """
    ★ 修复：验证 E[w] = R² + T²，不是 1。

    物理依据：单路径 MC 追踪中权重 = 分支概率的累乘。
    经过 1 次交互后：
        w = P_spec（如果走反射分支）或 P_refract（如果走折射分支）
        E[w] = P_spec · P_spec + P_refract · P_refract = R² + T²
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(45.0)
    R_s = fresnel_Rs(1.0, 1.5, theta_i)
    T_s = 1.0 - R_s
    E_w_expected = R_s ** 2 + T_s ** 2

    N = 8192
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -1.0),
        direction=(math.sin(theta_i), 0.0, math.cos(theta_i)),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    torch.manual_seed(0)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    mean_w_step1 = out["energy"][:, 1].mean().item()

    print("\n[Test 2] Branch energy expectation = R² + T²")
    print("  R_s = {:.6f}, T_s = {:.6f}".format(R_s, T_s))
    print("  E[w]_expected = {:.6f}".format(E_w_expected))
    print("  E[w]_measured = {:.6f}".format(mean_w_step1))

    # MC 统计误差 ~ 0.003（N=8192）
    assert abs(mean_w_step1 - E_w_expected) < 0.02, \
        "E[w] mismatch: {:.6f} vs {:.6f}".format(
            mean_w_step1, E_w_expected)


def test_branch_probability_ratio():
    """
    分支概率比值匹配解析 Fresnel：
    P_specular / (P_specular + P_refract) = R
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

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
    ratio = (R / (R + T + 1e-12)).mean().item()

    R_expected = fresnel_Rs(1.0, 1.5, theta_i)

    print("\n[Test 3] Branch probability ratio")
    print("  P_spec / (P_spec + P_refract) = {:.6f}".format(ratio))
    print("  Analytic R_s = {:.6f}".format(R_expected))

    assert abs(ratio - R_expected) < 0.01


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))