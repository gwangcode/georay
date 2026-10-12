"""RK4 integrators for GRIN media."""
import torch


def rk4_step_grin(r, d, grin_field, wavelengths, ds):
    """Single RK4 step. ds can be scalar, 0-dim, or [K] tensor."""
    if isinstance(ds, torch.Tensor):
        if ds.dim() == 0:
            ds_col = ds
        elif ds.dim() == 1:
            ds_col = ds.unsqueeze(-1)
        else:
            ds_col = ds
    else:
        ds_col = ds

    n_curr, grad_n_curr = grin_field(r, wavelengths)
    p = n_curr * d
    dr1 = p / n_curr
    dp1 = grad_n_curr

    r_k2 = r + 0.5 * ds_col * dr1
    p_k2 = p + 0.5 * ds_col * dp1
    n_k2, grad_n_k2 = grin_field(r_k2, wavelengths)
    dr2 = p_k2 / n_k2
    dp2 = grad_n_k2

    r_k3 = r + 0.5 * ds_col * dr2
    p_k3 = p + 0.5 * ds_col * dp2
    n_k3, grad_n_k3 = grin_field(r_k3, wavelengths)
    dr3 = p_k3 / n_k3
    dp3 = grad_n_k3

    r_k4 = r + ds_col * dr3
    p_k4 = p + ds_col * dp3
    n_k4, grad_n_k4 = grin_field(r_k4, wavelengths)
    dr4 = p_k4 / n_k4
    dp4 = grad_n_k4

    r_next = r + (ds_col / 6.0) * (dr1 + 2 * dr2 + 2 * dr3 + dr4)
    p_next = p + (ds_col / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
    n_next, _ = grin_field(r_next, wavelengths)
    d_next = p_next / n_next
    d_next = d_next / torch.norm(d_next, dim=-1, keepdim=True)
    return r_next, d_next, n_next


def adaptive_rk4_step_per_ray(r, d, grin_field, wavelengths, h_per_ray,
                              tol=1e-4, h_min=1e-4, h_max=0.2):
    """Per-ray adaptive RK4 with anti-deadlock protection."""
    r1, d1, n1 = rk4_step_grin(r, d, grin_field, wavelengths, h_per_ray)
    h_half = h_per_ray * 0.5
    r_mid, d_mid, _ = rk4_step_grin(r, d, grin_field, wavelengths, h_half)
    r2, d2, n2 = rk4_step_grin(r_mid, d_mid, grin_field, wavelengths, h_half)

    err = torch.maximum(torch.norm(r1 - r2, dim=-1), torch.norm(d1 - d2, dim=-1))
    accepted = err < tol
    at_h_min = h_per_ray <= (h_min * 1.05)
    force_accept = (~accepted) & at_h_min
    accepted = accepted | force_accept

    acc_3d = accepted.unsqueeze(-1)
    r_next = torch.where(acc_3d, r2, r)
    d_next = torch.where(acc_3d, d2, d)

    # ★ 修复：删掉 no_grad（CustomFunctionGRINField 内部用 autograd.grad）
    n_curr_tmp, _ = grin_field(r, wavelengths)
    n_next = torch.where(acc_3d, n2, n_curr_tmp)

    h_max_t = torch.full_like(h_per_ray, float(h_max))
    h_min_t = torch.full_like(h_per_ray, float(h_min))
    new_h = torch.where(accepted,
                        torch.minimum(h_per_ray * 1.5, h_max_t),
                        torch.maximum(h_per_ray * 0.5, h_min_t))
    return r_next, d_next, n_next, new_h, err, accepted