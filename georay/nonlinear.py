"""Nonlinear optics: Kerr self-focusing helper."""
import numpy as np


def kerr_index(n0, n2, intensity):
    """Kerr nonlinear refractive index: n(I) = n0 + n2 * I."""
    return n0 + n2 * intensity


def self_focusing_critical_power(wavelength_nm, n0, n2):
    """
    Kerr self-focusing critical power: P_cr = 3.77 * lambda^2 / (8*pi*n0*n2).

    Args:
        wavelength_nm: wavelength (nm).
        n0: linear refractive index.
        n2: nonlinear refractive index (m^2/W).

    Returns:
        P_cr (W).
    """
    lam_m = wavelength_nm * 1e-9
    return 3.77 * lam_m ** 2 / (8.0 * np.pi * n0 * n2)
