"""Thin-film interference: transfer matrix method for multilayer stacks."""
import numpy as np


def transfer_matrix_stack(wavelength_nm, theta_i_rad, n_list, d_list,
                           polarization="s"):
    """
    计算多层薄膜的反射率 R 和透射率 T。

    Args:
        wavelength_nm: 真空波长 (nm)
        theta_i_rad: 入射角 (弧度)
        n_list: [n0, n1, ..., nN, n_sub] 各层折射率
        d_list: [d1, ..., dN] 各层物理厚度 (nm)
        polarization: "s" 或 "p"

    Returns:
        (R, T): 反射率、透射率
    """
    n = np.asarray(n_list, dtype=complex)
    d = np.asarray(d_list, dtype=float)

    sin_i = np.sin(theta_i_rad)
    cos_theta = np.sqrt(1.0 - (n[0] * sin_i / n) ** 2 + 0j)

    if polarization == "s":
        eta = n * cos_theta
    else:
        eta = n / cos_theta

    M = np.eye(2, dtype=complex)
    for j in range(len(d)):
        delta = 2 * np.pi * n[j + 1] * d[j] * cos_theta[j + 1] / wavelength_nm
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        m_j = np.array([
            [cos_d, 1j * sin_d / eta[j + 1]],
            [1j * sin_d * eta[j + 1], cos_d]
        ])
        M = M @ m_j

    B = M[0, 0] + M[0, 1] * eta[-1]
    C = M[1, 0] + M[1, 1] * eta[-1]
    r = (eta[0] * B - C) / (eta[0] * B + C)
    t = 2 * eta[0] / (eta[0] * B + C)

    R = float(abs(r) ** 2)
    T = float(np.real(eta[-1]) / np.real(eta[0]) * abs(t) ** 2)
    return R, T


def quarter_wave_ar_coating(n_substrate=1.5, wavelength_nm=550.0):
    """四分之一波长增透膜：n_coating = sqrt(n_substrate)。"""
    n_coating = float(np.sqrt(n_substrate))
    d = wavelength_nm / (4.0 * n_coating)
    return n_coating, d