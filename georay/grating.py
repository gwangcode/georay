"""Diffraction grating equations."""
import numpy as np


def grating_angle(wavelength_nm, period_nm, theta_i_rad, order):
    """
    光栅方程：sin(θm) = m·λ/d - sin(θi)

    Returns:
        (theta_m_rad, is_valid): 出射角，是否有效（|sin θm| ≤ 1）
    """
    sin_m = order * wavelength_nm / period_nm - np.sin(theta_i_rad)
    if abs(sin_m) > 1.0:
        return None, False
    return float(np.arcsin(sin_m)), True


def angular_dispersion(wavelength_nm, period_nm, theta_i_rad, order):
    """角色散 dθ/dλ (rad/nm) 的解析值。"""
    theta_m, valid = grating_angle(wavelength_nm, period_nm, theta_i_rad, order)
    if not valid:
        return None
    cos_m = np.cos(theta_m)
    return float(order / (period_nm * cos_m))