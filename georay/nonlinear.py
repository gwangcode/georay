"""Nonlinear optics: Kerr self-focusing helper."""
import numpy as np


def kerr_index(n0, n2, intensity):
    """Kerr 非线性折射率：n(I) = n0 + n2·I。"""
    return n0 + n2 * intensity


def self_focusing_critical_power(wavelength_nm, n0, n2):
    """
    Kerr 自聚焦临界功率：P_cr = 3.77·λ² / (8π·n0·n2)

    Args:
        wavelength_nm: 波长 (nm)
        n0: 线性折射率
        n2: 非线性折射率 (m²/W)

    Returns:
        P_cr (W)
    """
    lam_m = wavelength_nm * 1e-9
    return 3.77 * lam_m ** 2 / (8.0 * np.pi * n0 * n2)