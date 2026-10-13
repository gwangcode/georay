"""
tests/test_nonlinear_optics.py
===============================
Kerr 非线性光学的基础验证。
"""
import math
import pytest

from georay.nonlinear import kerr_index, self_focusing_critical_power


def test_kerr_index_linear_regime():
    """低光强下 n ≈ n0。"""
    n0 = 1.5
    n2 = 1e-20
    I = 1e10  # W/m²

    n = kerr_index(n0, n2, I)
    print("\n[Test 1] Kerr index at low intensity")
    print("  n(I) = {:.10f}".format(n))

    assert abs(n - n0) < 1e-9


def test_kerr_index_high_intensity():
    """高光强下 n 显著变化。"""
    n0 = 1.5
    n2 = 1e-18
    I = 1e16

    n = kerr_index(n0, n2, I)
    print("\n[Test 2] Kerr index at high intensity")
    print("  n(I) = {:.6f}".format(n))

    assert n > n0
    assert abs(n - (n0 + n2 * I)) < 1e-9


def test_self_focusing_critical_power():
    """自聚焦临界功率的解析式验证。"""
    wl = 800.0
    n0 = 1.5
    n2 = 2.6e-20   # 典型 Kerr 系数

    P_cr = self_focusing_critical_power(wl, n0, n2)

    # 解析式：P_cr = 3.77 λ² / (8π n0 n2)
    lam_m = wl * 1e-9
    P_expected = 3.77 * lam_m ** 2 / (8 * math.pi * n0 * n2)

    print("\n[Test 3] Self-focusing critical power")
    print("  P_cr = {:.4e} W".format(P_cr))
    print("  Expected: {:.4e} W".format(P_expected))

    assert abs(P_cr - P_expected) < 1e-15
    # 典型值应在 MW 量级
    assert 1e6 < P_cr < 1e8


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))