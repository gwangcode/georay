"""
tests/test_thin_film.py
========================
薄膜干涉验证：单层增透膜、多层高反膜、与解析 Fresnel 一致性。
"""
import math
import numpy as np
import pytest

from georay.thin_film import transfer_matrix_stack, quarter_wave_ar_coating


# ==============================================================================
# Test 1: 无膜 → 与 Fresnel 公式一致
# ==============================================================================

def test_no_coating_matches_fresnel():
    """d=0 时应退化为单界面 Fresnel。"""
    n1, n2 = 1.0, 1.5
    wl = 550.0
    theta = math.radians(30.0)

    R_tmm, T_tmm = transfer_matrix_stack(wl, theta, [n1, n2], [])

    # 解析 Fresnel
    sin_t = n1 * math.sin(theta) / n2
    cos_t = math.sqrt(1 - sin_t ** 2)
    cos_i = math.cos(theta)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    R_s = r_s ** 2

    print("\n[Test 1] No coating matches Fresnel")
    print("  R_tmm = {:.6f}, R_fresnel = {:.6f}".format(R_tmm, R_s))

    assert abs(R_tmm - R_s) < 1e-6
    assert abs(R_tmm + T_tmm - 1.0) < 1e-6


# ==============================================================================
# Test 2: 四分之一波长增透膜 → R ≈ 0
# ==============================================================================

def test_quarter_wave_ar_coating_zero_reflection():
    """
    在玻璃表面加一层 λ/4 增透膜（n_coating = sqrt(n_sub)），
    垂直入射时反射率应接近 0。
    """
    n_sub = 1.5
    wl = 550.0
    n_coating, d = quarter_wave_ar_coating(n_sub, wl)

    R, T = transfer_matrix_stack(wl, 0.0, [1.0, n_coating, n_sub], [d])

    print("\n[Test 2] Quarter-wave AR coating")
    print("  n_coating = {:.4f}, d = {:.2f} nm".format(n_coating, d))
    print("  R = {:.8f}, T = {:.8f}".format(R, T))

    assert R < 1e-4, "AR coating should give near-zero reflection"
    assert abs(R + T - 1.0) < 1e-6


# ==============================================================================
# Test 3: 多层高反膜
# ==============================================================================

def test_multilayer_high_reflector():
    """
    交替高低折射率的多层膜堆，反射率应随层数增加。
    """
    n_H, n_L = 2.5, 1.5
    wl = 550.0
    d_H = wl / (4 * n_H)
    d_L = wl / (4 * n_L)

    R_prev = 0.0
    for n_pairs in [1, 3, 5]:
        n_list = [1.0]
        d_list = []
        for _ in range(n_pairs):
            n_list.append(n_H)
            d_list.append(d_H)
            n_list.append(n_L)
            d_list.append(d_L)
        n_list.append(1.5)

        R, _ = transfer_matrix_stack(wl, 0.0, n_list, d_list)
        print("\n[Test 3] High reflector: {} pairs, R = {:.6f}".format(
            n_pairs, R))
        assert R > R_prev, "Reflectance should increase with more pairs"
        R_prev = R


# ==============================================================================
# Test 4: 2λ 厚度的增透膜（无增透效果）
# ==============================================================================

def test_double_thickness_no_effect():
    """厚度为 λ/2 时，薄膜不改变反射率（与无膜相同）。"""
    n_sub = 1.5
    wl = 550.0
    n_coating = math.sqrt(n_sub)
    d = wl / (2 * n_coating)

    R_coated, _ = transfer_matrix_stack(wl, 0.0, [1.0, n_coating, n_sub], [d])
    R_bare, _ = transfer_matrix_stack(wl, 0.0, [1.0, n_sub], [])

    print("\n[Test 4] Double thickness (λ/2)")
    print("  R_coated = {:.6f}".format(R_coated))
    print("  R_bare   = {:.6f}".format(R_bare))

    assert abs(R_coated - R_bare) < 1e-4


# ==============================================================================
# Test 5: 能量守恒（无吸收）
# ==============================================================================

def test_energy_conservation():
    """无吸收薄膜：R + T = 1（各角度、各偏振）。"""
    wl = 550.0
    n_list = [1.0, 1.38, 1.5, 1.7, 2.5, 1.5]
    d_list = [100.0, 80.0, 60.0, 40.0, 20.0]

    for pol in ["s", "p"]:
        for theta_deg in [0.0, 30.0, 60.0]:
            theta = math.radians(theta_deg)
            R, T = transfer_matrix_stack(wl, theta, n_list, d_list, pol)
            err = abs(R + T - 1.0)
            print("\n[Test 5] pol={}, θ={}°: R={:.6f}, T={:.6f}, |R+T-1|={:.2e}".format(
                pol, theta_deg, R, T, err))
            assert err < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))