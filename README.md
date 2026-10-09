# GeoRay

[![DOI](https://zenodo.org/badge/1389842626.svg)](https://doi.org/10.5281/zenodo.22983753)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

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
git clone https://github.com/gwangcode/GeoRay.git
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
## Examples
Three runnable demos are provided:

```bash
python examples/demo_cube.py           # Fresnel reflection at cube surfaces
python examples/demo_grin.py           # Ray bending in a parabolic GRIN medium
python examples/demo_polarization.py   # s/p polarization evolution at a glass interface
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
@article{yourname2026georay,
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
