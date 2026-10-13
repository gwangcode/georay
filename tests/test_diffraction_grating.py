"""
tests/test_diffraction_grating.py
==================================
衍射光栅方程的验证。
"""
import math
import pytest

from georay.grating import grating_angle, angular_dispersion


def test_grating_equation_zero_order():
    """零级衍射方向等于镜面反射方向。"""
    wl = 550.0
    period = 1000.0
    theta_i = math.radians(30.0)

    theta_0, valid = grating_angle(wl, period, theta_i, 0)
    assert valid

    print("\n[Test 1] Zero-order diffraction")
    print("  θi = 30°, θ_0 = {:.4f}°".format(math.degrees(theta_0)))

    assert abs(math.degrees(theta_0) - math.degrees(theta_i)) < 1e-4


def test_grating_equation_first_order():
    """一级衍射角满足解析公式。"""
    wl = 550.0
    period = 2000.0
    theta_i = math.radians(20.0)

    theta_1, valid = grating_angle(wl, period, theta_i, 1)
    assert valid

    sin_expected = wl / period - math.sin(theta_i)
    theta_expected = math.asin(sin_expected)

    print("\n[Test 2] First-order diffraction")
    print("  θi = 20°, θ_1 = {:.4f}°".format(math.degrees(theta_1)))
    print("  Expected: {:.4f}°".format(math.degrees(theta_expected)))

    assert abs(theta_1 - theta_expected) < 1e-6


def test_grating_angular_dispersion():
    """角色散与解析解一致。"""
    wl = 550.0
    period = 1000.0
    theta_i = 0.0

    disp = angular_dispersion(wl, period, theta_i, 1)
    theta_1, _ = grating_angle(wl, period, theta_i, 1)
    disp_expected = 1.0 / (period * math.cos(theta_1))

    print("\n[Test 3] Angular dispersion")
    print("  dθ/dλ = {:.6e} rad/nm".format(disp))
    print("  Expected: {:.6e}".format(disp_expected))

    assert abs(disp - disp_expected) < 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))