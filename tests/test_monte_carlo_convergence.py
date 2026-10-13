"""
tests/test_monte_carlo_convergence.py
======================================
Monte Carlo 追踪的统计收敛性验证。

物理依据：
    估计量 ⟨R⟩ 的标准误差 ∝ 1/sqrt(N)。
    增加光线数 N，平均反射率应收敛到解析 Fresnel 值。
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
# Test 1: 平均反射率收敛到解析 Fresnel
# ==============================================================================

def test_mc_convergence_to_analytic_fresnel():
    """
    N 增大时，Monte Carlo 平均反射率应收敛到解析 Fresnel 值。

    用 s 偏振 + 45° 入射角，期望 R_s ≈ 0.092。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(45.0)
    R_s_expected, _ = fresnel_rs_rp(1.0, 1.5, theta_i)

    # 不同 N 下的统计均值
    N_list = [100, 1000, 10000]
    errors = []

    print("\n[Test 1] MC convergence to analytic Fresnel")
    print("  θi = 45°, n1 = 1.0, n2 = 1.5")
    print("  R_s analytic = {:.6f}".format(R_s_expected))
    print("  {:>8s} {:>14s} {:>14s}".format("N", "mean(R)", "|err|"))

    for N in N_list:
        rays = generate_rays(
            source_type="grid", num_rays=N,
            origin=(0.0, 0.0, -1.0),
            direction=(math.sin(theta_i), 0.0, math.cos(theta_i)),
            bounds=(-0.5, 0.5, -0.5, 0.5),
            polarization="s", device=device)

        # 用固定 seed 保证可重复性
        torch.manual_seed(42)
        tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

        R_mean = out["reflectivity"][:, 1, 0].mean().item()
        err = abs(R_mean - R_s_expected)
        errors.append(err)
        print("  {:>8d} {:>14.6f} {:>14.2e}".format(N, R_mean, err))

    # 收敛性：N 增大 10 倍，误差应减小（非单调但总体趋势）
    assert errors[-1] < errors[0], \
        "Error should decrease with more samples: {} -> {}".format(
            errors[0], errors[-1])


# ==============================================================================
# Test 2: 1/sqrt(N) 收敛速率
# ==============================================================================

def test_mc_convergence_rate():
    """
    Monte Carlo 估计误差应 ~ 1/sqrt(N)。
    N 增大 100 倍，误差应减小约 10 倍。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(60.0)
    R_s_expected, _ = fresnel_rs_rp(1.0, 1.5, theta_i)

    # 用多个 seed 平均，减少单个 seed 的随机性
    n_seeds = 5
    N_small, N_large = 100, 10000

    def mean_R(N):
        values = []
        for seed in range(n_seeds):
            rays = generate_rays(
                source_type="grid", num_rays=N,
                origin=(0.0, 0.0, -1.0),
                direction=(math.sin(theta_i), 0.0, math.cos(theta_i)),
                bounds=(-0.5, 0.5, -0.5, 0.5),
                polarization="s", device=device)
            torch.manual_seed(seed)
            tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
            out = tracer.trace(rays, max_steps=3, surface_model="fresnel")
            values.append(out["reflectivity"][:, 1, 0].mean().item())
        return sum(values) / len(values)

    err_small = abs(mean_R(N_small) - R_s_expected)
    err_large = abs(mean_R(N_large) - R_s_expected)

    ratio_N = N_large / N_small  # 100
    expected_err_ratio = math.sqrt(ratio_N)  # 10
    actual_err_ratio = err_small / err_large if err_large > 0 else float("inf")

    print("\n[Test 2] MC convergence rate (1/sqrt(N))")
    print("  N_small = {}, N_large = {}".format(N_small, N_large))
    print("  err(N_small) = {:.4e}".format(err_small))
    print("  err(N_large) = {:.4e}".format(err_large))
    print("  err ratio: actual={:.2f}, expected≈{:.2f}".format(
        actual_err_ratio, expected_err_ratio))

    # 允许 3 倍偏差
    assert actual_err_ratio > expected_err_ratio / 3.0, \
        "Convergence too fast (possible bias): ratio = {:.2f}".format(actual_err_ratio)


# ==============================================================================
# Test 3: 均值应落在 3σ 置信区间内
# ==============================================================================

def test_mc_mean_within_3sigma():
    """
    Monte Carlo 均值应落在解析值的 3σ 置信区间内。
    σ ≈ sqrt(p·(1-p)/N)，其中 p 是理论概率。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(30.0)
    R_s_expected, _ = fresnel_rs_rp(1.0, 1.5, theta_i)

    N = 5000
    n_trials = 10
    means = []

    for seed in range(n_trials):
        rays = generate_rays(
            source_type="grid", num_rays=N,
            origin=(0.0, 0.0, -1.0),
            direction=(math.sin(theta_i), 0.0, math.cos(theta_i)),
            bounds=(-0.5, 0.5, -0.5, 0.5),
            polarization="s", device=device)
        torch.manual_seed(seed)
        tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        out = tracer.trace(rays, max_steps=3, surface_model="fresnel")
        means.append(out["reflectivity"][:, 1, 0].mean().item())

    # 理论 σ（二项分布）
    sigma = math.sqrt(R_s_expected * (1 - R_s_expected) / N)
    global_mean = sum(means) / len(means)
    dev = abs(global_mean - R_s_expected)

    print("\n[Test 3] MC mean within 3σ confidence interval")
    print("  R_expected = {:.6f}".format(R_s_expected))
    print("  global mean = {:.6f}".format(global_mean))
    print("  σ (binomial) = {:.6f}".format(sigma))
    print("  |dev| = {:.6f}, 3σ = {:.6f}".format(dev, 3 * sigma))

    assert dev < 3 * sigma, \
        "Global mean deviates more than 3σ: {:.4e} > {:.4e}".format(
            dev, 3 * sigma)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))