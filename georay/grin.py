"""Gradient-index (GRIN) refractive index field definitions."""
import torch
import torch.nn as nn

from .interpolation import gpu_interp1d, gpu_interp2d


class BaseGRINField(nn.Module):
    """Base class for GRIN fields.

    Subclasses implement `forward(r, wavelength)` returning (n, grad_n).

    For anisotropic fields, `trace()` sets `self._current_pol` and
    `self._current_dop` before calling `forward()`. Isotropic fields can
    simply ignore these.
    """

    def __init__(self):
        super().__init__()
        self._current_pol = None
        self._current_dop = None

    def set_context(self, pol, dop):
        """Set polarization context. Called by trace(). pol: [N,3], dop: [N,1]."""
        self._current_pol = pol
        self._current_dop = dop

    def clear_context(self):
        self._current_pol = None
        self._current_dop = None

    def forward(self, r, wavelength=None):
        raise NotImplementedError


def _adaptive_gradient_scalar(field_eval, r, wavelength, eps_base=1e-3):
    """Adaptive Richardson-extrapolated gradient (4th order)."""
    scale = torch.clamp(torch.norm(r, dim=-1, keepdim=True), min=1.0)
    h = eps_base * scale
    grads = []
    for axis in range(3):
        e = torch.zeros_like(r)
        e[..., axis] = 1.0
        h_col = h
        r_p = r + h_col * e
        r_m = r - h_col * e
        r_ph = r + 0.5 * h_col * e
        r_mh = r - 0.5 * h_col * e
        n_p = field_eval(r_p, wavelength)
        n_m = field_eval(r_m, wavelength)
        n_ph = field_eval(r_ph, wavelength)
        n_mh = field_eval(r_mh, wavelength)
        g1 = (n_p - n_m) / (2.0 * h_col)
        g2 = (n_ph - n_mh) / h_col
        grads.append((4.0 * g2 - g1) / 3.0)
    return torch.cat(grads, dim=-1)


def parse_grin_input_to_field(grin_cfg, device):
    """Parse GRIN config (constant/callable/dict/BaseGRINField) into a field instance."""

    if isinstance(grin_cfg, BaseGRINField):
        return grin_cfg

    if callable(grin_cfg):
        class CustomFunctionGRINField(BaseGRINField):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def forward(self, r, wavelength=None):
                r_req = r.detach().clone().requires_grad_(True)
                try:
                    x, y, z = r_req[:, 0:1], r_req[:, 1:2], r_req[:, 2:3]
                    n = self.func(x, y, z, wavelength)
                except TypeError:
                    n = self.func(r_req, wavelength)
                if not isinstance(n, torch.Tensor):
                    n = torch.tensor(n, dtype=torch.float32, device=r.device)
                if n.dim() == 1:
                    n = n.unsqueeze(-1)
                grad_n = torch.autograd.grad(
                    outputs=n, inputs=r_req, grad_outputs=torch.ones_like(n),
                    create_graph=False, retain_graph=False, allow_unused=True
                )[0]
                if grad_n is None:
                    grad_n = torch.zeros_like(r)
                return n.detach(), grad_n.detach()

        return CustomFunctionGRINField(grin_cfg)

    if isinstance(grin_cfg, dict):
        class DictGridGRINField(BaseGRINField):
            def __init__(self, cfg, dev):
                super().__init__()
                self.device = dev
                self.has_wl = "wavelengths_nm" in cfg or "wl" in cfg
                self.wl_grid = torch.tensor(cfg.get("wavelengths_nm", cfg.get("wl", [550.0])),
                                            dtype=torch.float32, device=dev)
                self.x_grid = torch.tensor(cfg.get("x", [0.0]), dtype=torch.float32, device=dev)
                self.y_grid = torch.tensor(cfg.get("y", [0.0]), dtype=torch.float32, device=dev)
                self.z_grid = torch.tensor(cfg.get("z", [0.0]), dtype=torch.float32, device=dev)
                self.values = torch.tensor(cfg["values"], dtype=torch.float32, device=dev)

            def _trilinear(self, x, y, z, wl=None):
                ix = torch.clamp(torch.searchsorted(self.x_grid, x, right=False),
                                 1, self.x_grid.numel() - 1)
                tx = ((x - self.x_grid[ix - 1]) /
                      (self.x_grid[ix] - self.x_grid[ix - 1] + 1e-12)).clamp(0, 1)
                iy = torch.clamp(torch.searchsorted(self.y_grid, y, right=False),
                                 1, self.y_grid.numel() - 1)
                ty = ((y - self.y_grid[iy - 1]) /
                      (self.y_grid[iy] - self.y_grid[iy - 1] + 1e-12)).clamp(0, 1)
                iz = torch.clamp(torch.searchsorted(self.z_grid, z, right=False),
                                 1, self.z_grid.numel() - 1)
                tz = ((z - self.z_grid[iz - 1]) /
                      (self.z_grid[iz] - self.z_grid[iz - 1] + 1e-12)).clamp(0, 1)

                if self.has_wl and wl is not None:
                    iw = torch.clamp(torch.searchsorted(self.wl_grid, wl, right=False),
                                     1, self.wl_grid.numel() - 1)
                    tw = ((wl - self.wl_grid[iw - 1]) /
                          (self.wl_grid[iw] - self.wl_grid[iw - 1] + 1e-12)).clamp(0, 1)
                    v0000 = self.values[ix-1, iy-1, iz-1, iw-1]
                    v0001 = self.values[ix-1, iy-1, iz-1, iw]
                    v0010 = self.values[ix-1, iy-1, iz, iw-1]
                    v0011 = self.values[ix-1, iy-1, iz, iw]
                    v0100 = self.values[ix-1, iy, iz-1, iw-1]
                    v0101 = self.values[ix-1, iy, iz-1, iw]
                    v0110 = self.values[ix-1, iy, iz, iw-1]
                    v0111 = self.values[ix-1, iy, iz, iw]
                    v1000 = self.values[ix, iy-1, iz-1, iw-1]
                    v1001 = self.values[ix, iy-1, iz-1, iw]
                    v1010 = self.values[ix, iy-1, iz, iw-1]
                    v1011 = self.values[ix, iy-1, iz, iw]
                    v1100 = self.values[ix, iy, iz-1, iw-1]
                    v1101 = self.values[ix, iy, iz-1, iw]
                    v1110 = self.values[ix, iy, iz, iw-1]
                    v1111 = self.values[ix, iy, iz, iw]
                    v000 = v0000*(1-tw) + v0001*tw
                    v001 = v0010*(1-tw) + v0011*tw
                    v010 = v0100*(1-tw) + v0101*tw
                    v011 = v0110*(1-tw) + v0111*tw
                    v100 = v1000*(1-tw) + v1001*tw
                    v101 = v1010*(1-tw) + v1011*tw
                    v110 = v1100*(1-tw) + v1101*tw
                    v111 = v1110*(1-tw) + v1111*tw
                    v00 = v000*(1-tz) + v001*tz
                    v01 = v010*(1-tz) + v011*tz
                    v10 = v100*(1-tz) + v101*tz
                    v11 = v110*(1-tz) + v111*tz
                    v0 = v00*(1-ty) + v01*ty
                    v1 = v10*(1-ty) + v11*ty
                    return v0*(1-tx) + v1*tx
                else:
                    v000 = self.values[ix-1, iy-1, iz-1]
                    v001 = self.values[ix-1, iy-1, iz]
                    v010 = self.values[ix-1, iy, iz-1]
                    v011 = self.values[ix-1, iy, iz]
                    v100 = self.values[ix, iy-1, iz-1]
                    v101 = self.values[ix, iy-1, iz]
                    v110 = self.values[ix, iy, iz-1]
                    v111 = self.values[ix, iy, iz]
                    v00 = v000*(1-tz) + v001*tz
                    v01 = v010*(1-tz) + v011*tz
                    v10 = v100*(1-tz) + v101*tz
                    v11 = v110*(1-tz) + v111*tz
                    v0 = v00*(1-ty) + v01*ty
                    v1 = v10*(1-ty) + v11*ty
                    return v0*(1-tx) + v1*tx

            def _evaluate(self, r, wl):
                n = self._trilinear(r[:, 0:1], r[:, 1:2], r[:, 2:3], wl)
                return n.unsqueeze(-1) if n.dim() == 1 else n

            def forward(self, r, wavelength=None):
                n_center = self._evaluate(r, wavelength)
                grad_n = _adaptive_gradient_scalar(self._evaluate, r, wavelength, eps_base=1e-3)
                return n_center, grad_n

        return DictGridGRINField(grin_cfg, device)

    class ConstantGRINField(BaseGRINField):
        def __init__(self, val, dev):
            super().__init__()
            self.val = float(val)
            self.device = dev

        def forward(self, r, wavelength=None):
            n = torch.full((r.shape[0], 1), self.val, dtype=torch.float32, device=self.device)
            return n, torch.zeros_like(r)

    return ConstantGRINField(grin_cfg, device)