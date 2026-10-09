"""GPU-accelerated interpolation utilities."""
import torch


def gpu_interp1d(x_new, x_grid, y_grid):
    """1D linear interpolation on GPU."""
    num_pts = x_grid.shape[0]
    if num_pts == 1:
        return y_grid.expand_as(x_new)
    idx = torch.clamp(torch.searchsorted(x_grid, x_new, right=False), 1, num_pts - 1)
    x0, x1 = x_grid[idx - 1], x_grid[idx]
    y0, y1 = y_grid[idx - 1], y_grid[idx]
    t = (x_new - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)


def gpu_interp2d(x_new, y_new, x_grid, y_grid, z_grid):
    """2D bilinear interpolation on GPU."""
    idx_x = torch.clamp(torch.searchsorted(x_grid, x_new, right=False), 1, x_grid.shape[0] - 1)
    idx_y = torch.clamp(torch.searchsorted(y_grid, y_new, right=False), 1, y_grid.shape[0] - 1)
    x0, x1 = x_grid[idx_x - 1], x_grid[idx_x]
    y0, y1 = y_grid[idx_y - 1], y_grid[idx_y]
    tx = (x_new - x0) / (x1 - x0 + 1e-12)
    ty = (y_new - y0) / (y1 - y0 + 1e-12)
    z00 = z_grid[idx_x - 1, idx_y - 1]
    z01 = z_grid[idx_x - 1, idx_y]
    z10 = z_grid[idx_x, idx_y - 1]
    z11 = z_grid[idx_x, idx_y]
    return ((1 - tx) * (1 - ty) * z00 + (1 - tx) * ty * z01 +
            tx * (1 - ty) * z10 + tx * ty * z11)


def create_pure_torch_interpolated_cfg(prop_val, device):
    """Parse material property config into a callable interpolation function."""
    if isinstance(prop_val, dict):
        has_wl = ("wavelengths_nm" in prop_val) or ("x" in prop_val)
        has_cos = ("cos_theta" in prop_val) or ("y" in prop_val)
        if has_wl and has_cos and "values" in prop_val:
            x_pts = prop_val.get("wavelengths_nm", prop_val.get("x"))
            y_pts = prop_val.get("cos_theta", prop_val.get("y"))
            z_pts = prop_val["values"]
            x_grid = torch.tensor(x_pts, dtype=torch.float32, device=device)
            y_grid = torch.tensor(y_pts, dtype=torch.float32, device=device)
            z_grid = torch.tensor(z_pts, dtype=torch.float32, device=device)
            return lambda x_in, y_in=None: (
                gpu_interp2d(x_in, y_in, x_grid, y_grid, z_grid)
                if y_in is not None else gpu_interp1d(x_in, x_grid, z_grid[:, 0])
            )
        else:
            x_pts = prop_val.get("wavelengths_nm",
                                 prop_val.get("x", prop_val.get("cos_theta", [])))
            y_pts = prop_val.get("values",
                                 prop_val.get("y", prop_val.get("val", [])))
            x_tensor = torch.tensor(x_pts, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y_pts, dtype=torch.float32, device=device)
            return lambda x_in, y_in=None: gpu_interp1d(x_in, x_tensor, y_tensor)
    elif callable(prop_val):
        return prop_val
    else:
        val_tensor = torch.tensor(float(prop_val), dtype=torch.float32, device=device)
        return lambda x_in, y_in=None: val_tensor.expand_as(x_in)