"""
tests/test_monte_carlo_convergence.py
======================================
Monte Carlo 追踪的正确性验证。

重要 API 语义：
    tracer 输出的 `reflectivity` / `transmissivity` / `absorptivity`
    是**界面处的理论概率**（P_specular / P_refract / P_absorb），
    与光线数 N 无关。它们对所有光线相同（给定固定几何）。

    因此无法通过统计 `reflectivity` 来验证 MC 收敛。
    正确的验证是：
      * 界面概率归一化：R + T + A = 1
      * 概率与解析 Fresnel 公式一致
      * 能量权重的期望符合理论分支概率
"""
import math
import torch
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_flat_interface(device, hs=100.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def fresnel_rs_rp(n1, n2, theta_i):
    sin_t = n1 * math.sin(theta_i) / n2
    if sin_t >= 1.0:
        return 1.0, 1.0
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return r_s ** 2, r_p ** 2


# ==============================================================================
# Test 1: 理论 reflectivity 字段精确匹配解析 Fresnel
# ==============================================================================

def test_theoretical_reflectivity_matches_analytic():
    """
    tracer 的 `reflectivity` 字段是界面处理论 P_specular，
    应与解析 Fresnel 公式精确匹配（误差 < 1e-6）。

    ★ 修复：原测试期望统计收敛，但该字段是理论值，与 N 无关。
       改为直接验证与解析值的一致性。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    print("\n[Test 1] Theoretical reflectivity matches analytic Fresnel")
    print("  {:>8s} {:>8s} {:>14s} {:>14s}".format(
        "θi", "pol", "R_analytic", "R_theory"))

    for theta_deg in [0.0, 30.0, 45.0, 60.0, 80.0]:
        theta_i = math.radians(theta_deg)
        R_s, R_p = fresnel_rs_rp(1.0, 1.5, theta_i)

        for pol, R_expected in [("s", R_s), ("p", R_p)]:
            rays = generate_rays(
                source_type="grid", num_rays=16,
                origin=(0.0, 0.0, -1.0),
                direction=(math.sin(theta_i), 0.0, math.cos(theta_i)),
                bounds=(-0.5, 0.5, -0.5, 0.5),
                polarization=pol, device=device)

            tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
            out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

            R_theory = out["reflectivity"][:, 1, 0].mean().item()
            print("  {:>8.1f} {:>8s} {:>14.6f} {:>14.6f}".format(
                theta_deg, pol, R_expected, R_theory))

            assert abs(R_theory - R_expected) < 1e-5, \
                "θ={}°, pol={}: R_theory={:.6f}, R_analytic={:.6f}".format(
                    theta_deg, pol, R_theory, R_expected)


# ==============================================================================
# Test 2: 界面概率归一化
# ==============================================================================

def test_interface_probability_normalization():
    """
    界面处 R + T + A = 1（概率归一化）。
    这个约束与解析 Fresnel 无关，只依赖于 tracer 内部逻辑。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.05},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.05},
    }

    theta_i = math.radians(45.0)
    rays = generate_rays(
        source_type="grid", num_rays=32,
        origin=(0.0, 0.0, -1.0),
        direction=(math.sin(theta_i), 0.0, math.cos(theta_i)),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    R = out["reflectivity"][:, 1, 0]
    T = out["transmissivity"][:, 1, 0]
    A = out["absorptivity"][:, 1, 0]
    total = R + T + A

    max_dev = (total - 1.0).abs().max().item()
    print("\n[Test 2] Interface probability normalization")
    print("  max |R + T + A - 1| = {:.3e}".format(max_dev))

    assert max_dev < 1e-5


# ==============================================================================
# Test 3: 能量权重的统计期望符合理论
# ==============================================================================

def test_energy_weight_expectation():
    """
    单次交互后，能量权重的期望：
        E[w] = P_spec · P_spec + P_refract · P_refract
             = R² + T²

    这是 Monte Carlo 单路径追踪的本质。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(45.0)
    R_s, _ = fresnel_rs_rp(1.0, 1.5, theta_i)
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

    # 第 1 步交互后的平均权重
    mean_w_step1 = out["energy"][:, 1].mean().item()

    print("\n[Test 3] Energy weight expectation")
    print("  R_s = {:.6f}, T_s = {:.6f}".format(R_s, T_s))
    print("  E[w]_expected = R² + T² = {:.6f}".format(E_w_expected))
    print("  E[w]_measured = {:.6f}".format(mean_w_step1))

    # Monte Carlo 统计误差：σ/√N ≈ 0.3/sqrt(8192) ≈ 0.003
    assert abs(mean_w_step1 - E_w_expected) < 0.02, \
        "E[w] mismatch: {:.6f} vs {:.6f}".format(mean_w_step1, E_w_expected)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))