"""Polarization utilities (Jones-Mueller matrices, Stokes vectors).

GPU optimization notes:
    * `_choose_perpendicular` uses in-place operations and `.where` to
      reduce temporary tensor allocations.
    * `rotate_stokes_reference` avoids unnecessary intermediate quantities.

Physics fix (v1.2):
    * `_choose_perpendicular` uses (1, 0, 0) as the reference axis when
      |d.z| > 0.99, instead of (0, 1, 0). This ensures the s-direction
      remains along the y-axis (true s) at near-normal incidence
      (theta_i < ~8 degrees), rather than falling into the plane of
      incidence and becoming the p-direction.
"""
import torch


def _choose_perpendicular(d):
    """Physical s-polarization direction = cross(ref, d).

    v1.2 fix:
        When |d.z| > 0.99 (near-normal incidence), use (1, 0, 0) as the
        reference axis instead of (0, 1, 0).

        Reason:
            d ~ (sin(theta), 0, cos(theta)) with small theta:
            cross((1, 0, 0), d) ~ (0, -cos(theta), 0) -> y-axis (true s)
            cross((0, 1, 0), d) ~ (cos(theta), 0, -sin(theta)) -> in-plane
            (p-direction, incorrect)

        This ensures the s-direction remains perpendicular to the plane
        of incidence (y-axis) across the full range of incidence angles,
        consistent with the s/p definitions in the Fresnel equations.
    """
    device = d.device

    # Default reference axis: z-axis
    ref = torch.zeros_like(d)
    ref[..., 2] = 1.0

    # Switch to x-axis near z
    near_z = d[..., 2].abs() > 0.99
    ref[near_z] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=d.dtype)

    s = torch.cross(ref, d, dim=-1)
    norm = s.norm(dim=-1, keepdim=True)
    eps = 1e-12
    safe_norm = norm.clamp_min(eps)
    s_normalized = s / safe_norm

    # Fallback: when d is parallel to ref (rare), use x-axis
    fallback = torch.zeros_like(s)
    fallback[..., 0] = 1.0
    degenerate = (norm < eps)
    return torch.where(degenerate, fallback, s_normalized)


def rotate_stokes_reference(stokes, s_old, s_new, d):
    """Rotate Stokes vector from (s_old, p_old) basis to (s_new, p_new) basis.

    Only S1 / S2 (linear polarization components) are rotated.
    S0 and S3 (circular polarization components) are invariant.
    """
    cos_t = torch.sum(s_old * s_new, dim=-1, keepdim=True)
    cross_val = torch.cross(s_old, s_new, dim=-1)
    sin_t = torch.sum(cross_val * d, dim=-1, keepdim=True)
    cos_2t = cos_t * cos_t - sin_t * sin_t
    sin_2t = 2.0 * cos_t * sin_t
    S0, S1, S2, S3 = (stokes[..., 0:1], stokes[..., 1:2],
                       stokes[..., 2:3], stokes[..., 3:4])
    S1_new = cos_2t * S1 + sin_2t * S2
    S2_new = -sin_2t * S1 + cos_2t * S2
    return torch.cat([S0, S1_new, S2_new, S3], dim=-1)


def apply_jones_mueller(stokes, r_s, r_p):
    """Apply Jones matrix diag(r_s, r_p) to a Stokes vector (normalized to S0=1).

    Standard Mueller matrix:
        a = 0.5 * (r_s^2 + r_p^2)
        b = 0.5 * (r_s^2 - r_p^2)
        c = r_s * r_p

        S0' = a*S0 + b*S1
        S1' = b*S0 + a*S1
        S2' = c*S2
        S3' = c*S3

    Output is normalized to S0' = 1.
    """
    a = 0.5 * (r_s * r_s + r_p * r_p)
    b = 0.5 * (r_s * r_s - r_p * r_p)
    c = r_s * r_p
    S0, S1, S2, S3 = (stokes[..., 0:1], stokes[..., 1:2],
                       stokes[..., 2:3], stokes[..., 3:4])
    S0_new = a * S0 + b * S1
    S1_new = b * S0 + a * S1
    S2_new = c * S2
    S3_new = c * S3
    S0_safe = torch.clamp(S0_new, min=1e-12)
    return torch.cat([torch.ones_like(S1_new),
                      S1_new / S0_safe, S2_new / S0_safe, S3_new / S0_safe],
                     dim=-1)


def randomize_stokes_diffuse(stokes):
    """Fully randomized Stokes vector (unpolarized).

    After diffuse reflection, polarization is completely randomized:
    S0 = 1, S1 = S2 = S3 = 0.
    """
    return torch.cat([torch.ones_like(stokes[..., 0:1]),
                      torch.zeros_like(stokes[..., 1:4])], dim=-1)


def stokes_to_pol_vector(stokes, s_ref, p_ref):
    """Derive main polarization direction from Stokes vector.

    Principal axis direction: psi = 0.5 * atan2(S2, S1)
    pol = cos(psi) * s_ref + sin(psi) * p_ref
    """
    S1 = stokes[..., 1:2]
    S2 = stokes[..., 2:3]
    psi = 0.5 * torch.atan2(S2, S1)
    pol = torch.cos(psi) * s_ref + torch.sin(psi) * p_ref
    return pol / (torch.norm(pol, dim=-1, keepdim=True) + 1e-12)


def degree_of_polarization(stokes):
    """DoP = sqrt(S1^2 + S2^2 + S3^2) / S0."""
    S0 = stokes[..., 0:1]
    S1 = stokes[..., 1:2]
    S2 = stokes[..., 2:3]
    S3 = stokes[..., 3:4]
    return torch.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / (S0 + 1e-12)
