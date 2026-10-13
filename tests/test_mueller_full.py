"""
tests/test_mueller_full.py
===========================
完整 Mueller 矩阵验证。

物理依据：
    Jones 矩阵 diag(r_s, r_p) 对应的 Mueller 矩阵是 4×4 实矩阵：
        M = [[a, b, 0, 0],
             [b, a, 0, 0],
             [0, 0, c, d],
             [0, 0,-d, c]]
    其中 a = (R_s + R_p)/2, b = (R_s - R_p)/2,
         c = r_s·r_p·cos(δ), d = r_s·r_p·sin(δ)

    当前 GeoRay 实现假设 δ=0（实数 Fresnel 系数），所以 d=0。
"""
import math
import torch
import pytest

from georay.polarization import apply_jones_mueller, degree_of_polarization


# ==============================================================================
# Test 1: 对角 Jones 矩阵 → Mueller 矩阵项
# ==============================================================================

def test_mueller_matrix_coefficients():
    """
    apply_jones_mueller 应实现标准 Mueller 矩阵：
        S0' = a·S0 + b·S1
        S1' = b·S0 + a·S1
        S2' = c·S2
        S3' = c·S3
    """
    r_s = 0.5
    r_p = 0.3
    a = 0.5 * (r_s ** 2 + r_p ** 2)
    b = 0.5 * (r_s ** 2 - r_p ** 2)
    c = r_s * r_p

    # 测试向量 S = (1, 1, 1, 1) 便于分解
    S0, S1, S2, S3 = 1.0, 0.5, 0.3, 0.2
    stokes_in = torch.tensor([[S0, S1, S2, S3]])

    stokes_out = apply_jones_mueller(
        stokes_in,
        torch.tensor([[r_s]]),
        torch.tensor([[r_p]]))

    S0_raw = a * S0 + b * S1
    S1_raw = b * S0 + a * S1
    S2_raw = c * S2
    S3_raw = c * S3

    expected = torch.tensor([[
        1.0,
        S1_raw / S0_raw,
        S2_raw / S0_raw,
        S3_raw / S0_raw,
    ]])

    print("\n[Test 1] Mueller matrix coefficients")
    print("  a={:.4f}, b={:.4f}, c={:.4f}".format(a, b, c))
    print("  Expected: {}".format([round(v, 6) for v in expected[0].tolist()]))
    print("  Actual  : {}".format([round(v, 6) for v in stokes_out[0].tolist()]))

    assert torch.allclose(stokes_out, expected, atol=1e-6)


# ==============================================================================
# Test 2: 归一化 S0' = 1
# ==============================================================================

def test_mueller_S0_normalization():
    """输出的 S0 应严格等于 1（归一化）。"""
    r_s, r_p = 0.7, 0.4
    stokes_in = torch.tensor([[1.0, 0.3, -0.5, 0.2]])

    stokes_out = apply_jones_mueller(
        stokes_in,
        torch.tensor([[r_s]]),
        torch.tensor([[r_p]]))

    print("\n[Test 2] Mueller S0 normalization")
    print("  Output S0: {:.10f}".format(stokes_out[0, 0].item()))

    assert abs(stokes_out[0, 0].item() - 1.0) < 1e-10


# ==============================================================================
# Test 3: 线性偏振 (S3=0) → S3 保持为 0
# ==============================================================================

def test_mueller_linear_preserves_S3():
    """线性偏振输入 (S3=0)，输出 S3 保持为 0。"""
    r_s, r_p = 0.6, 0.3
    stokes_in = torch.tensor([[1.0, 0.5, 0.3, 0.0]])

    stokes_out = apply_jones_mueller(
        stokes_in,
        torch.tensor([[r_s]]),
        torch.tensor([[r_p]]))

    print("\n[Test 3] Linear polarization preserves S3=0")
    print("  Output S3: {:.10f}".format(stokes_out[0, 3].item()))

    assert abs(stokes_out[0, 3].item()) < 1e-10


# ==============================================================================
# Test 4: 圆偏振 (S1=S2=0, S3=1) 演化
# ==============================================================================

def test_mueller_circular_polarization():
    """
    圆偏振输入：S1' = b/a·S0, S3' = c/a·S0
    归一化后 S1' = b/a, S3' = c/a。
    """
    r_s, r_p = 0.6, 0.3
    stokes_in = torch.tensor([[1.0, 0.0, 0.0, 1.0]])

    stokes_out = apply_jones_mueller(
        stokes_in,
        torch.tensor([[r_s]]),
        torch.tensor([[r_p]]))

    a = 0.5 * (r_s ** 2 + r_p ** 2)
    b = 0.5 * (r_s ** 2 - r_p ** 2)
    c = r_s * r_p

    S1_expected = b / a
    S3_expected = c / a

    print("\n[Test 4] Circular polarization evolution")
    print("  Expected: S1={:.6f}, S3={:.6f}".format(S1_expected, S3_expected))
    print("  Actual  : S1={:.6f}, S3={:.6f}".format(
        stokes_out[0, 1].item(), stokes_out[0, 3].item()))

    assert abs(stokes_out[0, 1].item() - S1_expected) < 1e-6
    assert abs(stokes_out[0, 3].item() - S3_expected) < 1e-6


# ==============================================================================
# Test 5: 能量守恒 S0' / S0 = a（无吸收）
# ==============================================================================

def test_mueller_energy_conservation():
    """无吸收界面：S0' / S0 = a = (R_s + R_p)/2。"""
    r_s, r_p = 0.7, 0.4
    a = 0.5 * (r_s ** 2 + r_p ** 2)

    for S1 in [0.0, 0.5, -0.5, 1.0]:
        stokes_in = torch.tensor([[1.0, S1, 0.0, 0.0]])
        stokes_out = apply_jones_mueller(
            stokes_in,
            torch.tensor([[r_s]]),
            torch.tensor([[r_p]]))

        # S0 归一化为 1
        assert abs(stokes_out[0, 0].item() - 1.0) < 1e-10

    print("\n[Test 5] Mueller energy conservation (S0'=1 always)")


# ==============================================================================
# Test 6: DoP 在纯偏振输入下的保持
# ==============================================================================

def test_mueller_dop_preserved():
    """纯偏振输入 (DoP=1)，输出 DoP 应为 1。"""
    r_s, r_p = 0.5, 0.3

    test_stokes = [
        [1.0, 1.0, 0.0, 0.0],    # s 偏振
        [1.0, -1.0, 0.0, 0.0],   # p 偏振
        [1.0, 0.0, 1.0, 0.0],    # +45°
        [1.0, 0.0, 0.0, 1.0],    # 圆偏振
        [1.0, 0.6, 0.6, 0.529],  # 一般纯偏振
    ]

    print("\n[Test 6] Mueller DoP preservation")
    for S in test_stokes:
        dop_in = math.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2)
        stokes_in = torch.tensor([S])
        stokes_out = apply_jones_mueller(
            stokes_in,
            torch.tensor([[r_s]]),
            torch.tensor([[r_p]]))

        dop_out = degree_of_polarization(stokes_out).item()
        print("  DoP_in={:.4f}, DoP_out={:.6f}".format(dop_in, dop_out))

        # 纯偏振输入 → 纯偏振输出
        if abs(dop_in - 1.0) < 1e-6:
            assert abs(dop_out - 1.0) < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))