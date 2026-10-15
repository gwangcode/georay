"""Diffraction grating equations."""
import numpy as np


def grating_angle(wavelength_nm, period_nm, theta_i_rad, order):
    """
    Grating equation: sin(theta_m) = m * lambda / d - sin(theta_i).

    Returns:
        (theta_m_rad, is_valid): diffraction angle and validity flag
        (|sin(theta_m)| <= 1).
    """
    sin_m = order * wavelength_nm / period_nm - np.sin(theta_i_rad)
    if abs(sin_m) > 1.0:
        return None, False
    return float(np.arcsin(sin_m)), True


def angular_dispersion(wavelength_nm, period_nm, theta_i_rad, order):
    """Analytical angular dispersion d(theta)/d(lambda) in rad/nm."""
    theta_m, valid = grating_angle(wavelength_nm, period_nm, theta_i_rad, order)
    if not valid:
        return None
    cos_m = np.cos(theta_m)
    return float(order / (period_nm * cos_m))
