"""Polarization utilities (Jones-Mueller matrices, Stokes vectors)."""
import torch


def _choose_perpendicular(d):
    """Choose a perpendicular reference direction (s_ref) for direction d."""
    ref = torch.zeros_like(d)
    ref[:, 2] = 1.0
    near_z = torch.abs(d[:, 2]) > 0.99
    ref[near_z] = torch.tensor([0.0, 1.0, 0.0], device=d.device)
    proj = torch.sum(ref * d, dim=-1, keepdim=True)
    s = ref - proj * d
    return s / (torch.norm(s, dim=-1, keepdim=True) + 1e-12)


def rotate_stokes_reference(stokes, s_old, s_new, d):
    """Rotate Stokes vector from (s_old, p_old) basis to (s_new, p_new) basis."""
    cos_t = torch.sum(s_old * s_new, dim=-1, keepdim=True)
    cross_val = torch.cross(s_old, s_new, dim=-1)
    sin_t = torch.sum(cross_val * d, dim=-1, keepdim=True)
    cos_2t = cos_t * cos_t - sin_t * sin_t
    sin_2t = 2.0 * cos_t * sin_t
    S0, S1, S2, S3 = stokes[..., 0:1], stokes[..., 1:2], stokes[..., 2:3], stokes[..., 3:4]
    S1_new = cos_2t * S1 + sin_2t * S2
    S2_new = -sin_2t * S1 + cos_2t * S2
    return torch.cat([S0, S1_new, S2_new, S3], dim=-1)


def apply_jones_mueller(stokes, r_s, r_p):
    """Apply Jones matrix diag(r_s, r_p) to a Stokes vector (normalized to S0=1)."""
    a = 0.5 * (r_s * r_s + r_p * r_p)
    b = 0.5 * (r_s * r_s - r_p * r_p)
    c = r_s * r_p
    S0, S1, S2, S3 = stokes[..., 0:1], stokes[..., 1:2], stokes[..., 2:3], stokes[..., 3:4]
    S0_new = a * S0 + b * S1
    S1_new = b * S0 + a * S1
    S2_new = c * S2
    S3_new = c * S3
    S0_safe = torch.clamp(S0_new, min=1e-12)
    return torch.cat([torch.ones_like(S1_new),
                      S1_new / S0_safe, S2_new / S0_safe, S3_new / S0_safe], dim=-1)


def randomize_stokes_diffuse(stokes):
    """Fully randomized Stokes vector (unpolarized)."""
    return torch.cat([torch.ones_like(stokes[..., 0:1]),
                      torch.zeros_like(stokes[..., 1:4])], dim=-1)


def stokes_to_pol_vector(stokes, s_ref, p_ref):
    """Derive main polarization direction from Stokes vector."""
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