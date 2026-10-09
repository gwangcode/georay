# GeoRay

[![DOI](https://zenodo.org/badge/1389842626.svg)](https://doi.org/10.5281/zenodo.22983753)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Tests](https://github.com/gwangcode/georay/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/gwangcode/georay/actions/workflows/test.yml)

**GeoRay** is a pure-PyTorch, GPU-accelerated ray tracer supporting gradient-index (GRIN) media and full Stokes polarization tracking.

---

## Features

- **GPU acceleration** — All operations vectorized in PyTorch (CPU or CUDA).
- **BVH acceleration** — SoA flattened BVH with 4-triangle leaves; fully vectorized ray-mesh intersection.
- **GRIN support** — Constant, function-based, or grid-based refractive index fields with adaptive RK4 integration.
- **Polarization tracking** — Full Stokes vector with Jones-Mueller matrices and reference-frame rotation.
- **Adaptive step control** — Per-ray RK4 with local error estimation and anti-deadlock protection.
- **Multiple light sources** — Grid, random, point, area, and mesh surface emitters.
- **Energy conservation** — Built-in check on reflection + transmission + absorption.

---

## Installation

```bash
git clone https://github.com/gwangcode/georay.git
cd GeoRay
pip install -e .
```
## Requirements
Python >= 3.8

PyTorch >= 2.0

NumPy >= 1.21

Trimesh >= 3.20

GPU is optional but recommended for large scenes.

Quick Start
```python
import torch
import trimesh
from georay import AdvancedRayTracerPBRTGPU, generate_rays

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build a cube
cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
face_mats[:, 0] = 0   # outside: air
face_mats[:, 1] = 1   # inside: glass

# Materials
materials_cfg = {
    0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
}

# Fire 16 rays from a grid
rays = generate_rays(
    source_type="grid", num_rays=16,
    origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.5, 0.5, -0.5, 0.5),
    polarization="s", wavelength_nm=550.0, device=device,
)

# Trace
tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials_cfg, device)
out = tracer.trace(rays, max_steps=40, fresnel=True)

print(f"Energy conserved: {out['energy_conserved']}")
print(f"Output shape: {out['positions'].shape}")
```

# Output Description
```tracer.trace(rays, ...)``` returns a Python ```dict``` of PyTorch tensors containing the full history of every ray. Let $N$ be the number of rays and $S$ the number of recorded steps (initial state + each propagation step). All tensors are on the same device as the input rays.

|Key	|Shape	|Dtype	|Description|
|:---   |:---   |:---   |:--- |
|positions	|[N, S+1, 3]	|float32    |3D position of each ray at every step, in world coordinates (x, y, z).|
|stokes	|[N, S+1, 4]	|float32	|Full Stokes vector (S0, S1, S2, S3) at every step. S0 is normalized to 1 after each scattering event.|
|active	|[N, S+1]	|bool	|Whether the ray is still propagating at each step (False after escape or absorption).|
|wavelengths	|[N, 1]	|float32	|Wavelength of each ray in nanometers (constant along the trace).|
|reflectivity	|[N, S+1, 1]	|float32	|Reflection coefficient at the last surface interaction of each step (0 if no surface was hit).|
|transmissivity	|[N, S+1, 1]	|float32	|Transmission coefficient at the last surface interaction of each step.|
|absorptivity	|[N, S+1, 1]	|float32	|Absorption coefficient at the last surface interaction of each step.|
|n1	|[N, S+1, 1]	|float32	|Refractive index on the incident side of the last surface hit in this step.|
|n2	|[N, S+1, 1]	|float32	|Refractive index on the transmitted side of the last surface hit in this step.|
|polarization_deg	|[N, S+1, 1]	|float32	|Linear polarization angle (degrees), computed as 0.5 * atan2(S2, S1).|
|dop	|[N, S+1, 1]	|float32	|Degree of polarization sqrt(S1² + S2² + S3²) / S0, in [0, 1].|
|energy	|[N, S+1]	|float32	|Remaining energy per ray. Starts at 1.0, drops to 0 after absorption.|
|rk_error	|[N, S+1]	|float32	|Local truncation error estimate from the adaptive RK4 integrator.|
|ds_history	|[N, S+1]	|float32	|Step size used at each step (adaptively adjusted per ray).|
|energy_conserved	|bool	|—	|True if every ray satisfies 0 ≤ energy ≤ 1 at every step.|
## Common access patterns
```python
# Final position of every ray
final_positions = out["positions"][:, -1, :]           # [N, 3]

# Full trajectory of ray i
trajectory_i = out["positions"][i]                     # [S+1, 3]

# Reflectivity at the first surface hit
R_first = out["reflectivity"][:, 1, 0]                 # [N]

# Material transitions across all rays
n1_hist = out["n1"][:, :, 0]                           # [N, S+1]
n2_hist = out["n2"][:, :, 0]

# Check for internal interfaces (e.g. material A → material B)
internal = (n1_hist > 1.0) & (n2_hist > 1.0) & (n1_hist != n2_hist)
```
## Export Utilities
GeoRay ships with two exporters that accept the output dict directly.

### NPZ — full trace history (NumPy binary)
```python
from georay import save_trace_output_to_npz

save_trace_output_to_npz(out, "trace.npz")
```
saves every tensor from the output dict as a NumPy array in a single compressed .npz file:

```text
trace.npz
├── positions           [N, S+1, 3]
├── stokes              [N, S+1, 4]
├── active              [N, S+1]
├── wavelengths         [N, 1]
├── reflectivity        [N, S+1, 1]
├── transmissivity      [N, S+1, 1]
├── absorptivity        [N, S+1, 1]
├── n1                  [N, S+1, 1]
├── n2                  [N, S+1, 1]
├── polarization_deg    [N, S+1, 1]
├── dop                 [N, S+1, 1]
├── energy              [N, S+1]
├── rk_error            [N, S+1]
└── ds_history          [N, S+1]
```
#### Load it back:

```python
import numpy as np
data = np.load("trace.npz")
positions = data["positions"]   # ndarray [N, S+1, 3]
stokes    = data["stokes"]      # ndarray [N, S+1, 4]
```
**Use case:** post-processing, plotting, or statistical analysis in NumPy / SciPy / Pandas.

### STL — 3D ray tracks (mesh file)
```python
from georay import export_ray_history_to_stl

export_ray_history_to_stl(out, "ray_tracks.stl", radius=0.02, sides=6)
```
saves every ray segment as a small cylinder in a single STL mesh. Parameters:

|Argument	|Default	|Description|
|:---       |:---       |:---|
|filename	|"ray_tracks.stl"	|Output file path.|
|radius	|0.02	|Radius of each cylinder (in world units).|
|sides	|6	|Number of polygon sides per cylinder cross-section.|
**Use case:** visual inspection in MeshLab, Blender, ParaView, or any STL viewer.

**Note:** STL is a surface mesh format (no color, no per-ray metadata). For data analysis use NPZ.

## Examples
Several runnable demos are provided:

```bash
python examples/demo_cube.py               # Fresnel reflection at cube surfaces
python examples/demo_grin.py               # Ray bending in a parabolic GRIN medium
python examples/demo_polarization.py       # s/p polarization evolution at a glass interface
python examples/demo_stepmesh.py           # Multi-body STEP assembly (requires stepmesh)
python examples/quick_check_stepfiles.py   # Validation script for stepmesh integration
```
Each demo prints the traced ray count, energy conservation status, and output shapes.

## Testing
```bash
pip install pytest
pytest tests/
```
The test suite covers:

BVH ray-mesh intersection (hit and miss cases)

GRIN field parsing (constant and function-based)

Polarization utilities (perpendicular basis, Jones-Mueller identity, DoP bounds)

## API Overview
### Symbol	Description
- **AdvancedRayTracerPBRTGPU**	Main ray tracer class

- **BaseGRINField** Base class for GRIN refractive index fields

- **parse_grin_input_to_field**	Parse GRIN config (constant / callable / dict)

- **generate_rays**	Generate rays from various source types

- **save_trace_output_to_npz**	Save trace results to compressed NPZ

- **export_ray_history_to_stl**	Export ray tracks to STL
GRIN Field Definition

### You can supply a GRIN field in three ways:


1. Constant:

    ```python
    materials_cfg[1]["grin_field"] = 1.5
    ```

2. Python function:

    ```python
    def my_grin(x, y, z, wavelength):
        return 1.5 - 0.1 * (x ** 2 + y ** 2)
    
    materials_cfg[1]["grin_field"] = my_grin
    ```
3. Grid (dict):

    ```python
    materials_cfg[1]["grin_field"] = {
        "x": [0.0, 0.5, 1.0],
        "y": [0.0, 0.5, 1.0],
        "z": [0.0, 0.5, 1.0],
        "values": [[[...]]],  # 3D or 4D array
    }
    ```
### Anisotropic / Polarization-Aware Fields
For anisotropic media (e.g. birefringent or polarization-sensitive GRIN), subclass BaseGRINField and read self._current_pol / self._current_dop inside forward():

```python
from georay import BaseGRINField
import torch

class BirefringentField(BaseGRINField):
    def forward(self, r, wavelength=None):
        pol = self._current_pol   # [N, 3] or None
        dop = self._current_dop   # [N, 1] or None

        n_avg = 1.5
        delta_n = 0.01
        if pol is None:
            return torch.full((r.shape[0], 1), n_avg, device=r.device), torch.zeros_like(r)

        axis = torch.tensor([1.0, 0.0, 0.0], device=r.device)
        cos2 = (pol @ axis) ** 2
        n_eff = n_avg + dop.squeeze(-1) * delta_n * (cos2 - 0.5)
        grad_n = torch.zeros_like(r)
        return n_eff.unsqueeze(-1), grad_n
```

## Performance Notes
GPU acceleration is automatic when CUDA is available.

For large scenes, use mixed precision (AMP) in your training loop if embedding GeoRay in a differentiable pipeline.

BVH build time is O(F log F) in number of triangles; tracing scales as O(log F) per ray.

## Limitations
Triangle meshes only (no NURBS or implicit surfaces).

Diffuse scattering assumes fully randomized polarization.

Anisotropic fields must be user-supplied as BaseGRINField subclasses.

No wavelength-dependent dispersion inside GRIN fields (user must supply n(λ)).

## Citation
If you use GeoRay in your research, please cite:

```bibtex
@article{wang2026georay,
  title   = {GeoRay: A GPU-Accelerated Ray Tracer for Gradient-Index and Polarization Optics},
  author  = {Gang Wang},
  journal = {Journal of Data- and Knowledge-Integrated Simulation Science},
  year    = {2026},
  doi     = {10.5281/zenodo.22983753}
}
```

## License
MIT License. See LICENSE for details.

## Contributing
Issues and pull requests are welcome on the GitHub repository.
