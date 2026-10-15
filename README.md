# GeoRay

[![DOI](https://zenodo.org/badge/1389842626.svg)](https://doi.org/10.5281/zenodo.22983753)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Tests](https://github.com/gwangcode/georay/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/gwangcode/georay/actions/workflows/test.yml)

**A GPU-Accelerated Ray Tracer for GRIN and Polarization Optics**

GeoRay is a research-grade ray tracing library with full Stokes polarization tracking, gradient-index (GRIN) media support, and a Warp-based GPU kernel implementation that achieves **1000×+ speedup** over a pure PyTorch BVH.

---

## Features

### Physics

- **Fresnel reflection / refraction** with s/p polarization weighting
- **Total internal reflection (TIR)**
- **Stokes vector polarization tracking** via the Jones-Mueller formalism
- **Diffuse reflection** (cosine-weighted hemisphere sampling)
- **Absorption** (energy-weighted Monte Carlo)
- **Monte Carlo energy weight** accumulation
- **Multiple bounces** with per-ray path length control
- **Russian roulette** termination (optional)

### Materials

- **Scalar refractive index** `n`
- **Wavelength-dependent** `n(λ)`, `reflectivity(λ)`, `diffuse(λ)`, `absorptivity(λ)`
- **2D LUT** `n(λ, cos θ)` for angle- and wavelength-dependent properties
- **Surface models**: `"fresnel"` / `"custom"` / `"coated"`
- **Gradient-index (GRIN) media**:
  - Analytic parabolic field `n(r) = n₀ − A·(x² + y²)`
  - User-supplied `@wp.func` callable (compile-time inlined, zero overhead)
  - 3D grid-based field with trilinear interpolation and analytic gradient
  - 4D grid-based field `n(x, y, z, λ)` with quadrilinear interpolation

### Performance

- **`AdvancedRayTracerPBRTGPU`** — Pure PyTorch BVH, works everywhere (CPU / GPU)
- **`WarpTracer`** — Full Warp GPU kernel, 1000×+ faster on large batches

### I/O & Integration

- Save trace output to compressed NPZ
- Export ray tracks to STL
- STEP mesh adapter (`stepmesh.glue()` → GeoRay)

---

## Installation

### Basic install (PyTorch BVH only)

```bash
pip install git+https://github.com/gwangcode/georay
```

### With Warp GPU acceleration (recommended)

```bash
pip install "georay[warp] @ git+https://github.com/gwangcode/georay"
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0, `trimesh` ≥ 3.20, and for Warp: a CUDA GPU with Compute Capability ≥ 5.0 (T4, V100, A100, RTX series, etc.).

---

## Quick Start

### PyTorch tracer (full feature set)

```python
import torch, trimesh
from georay import AdvancedRayTracerPBRTGPU, generate_rays

device = torch.device("cuda")

# ---- Scene: 2×2×2 glass cube ----
cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
face_mats[:, 0] = 0   # +normal side → air
face_mats[:, 1] = 1   # −normal side → glass

materials = {
    0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
}

# ---- Rays: 1024 parallel rays through the cube ----
rays = generate_rays(
    source_type="grid", num_rays=1024,
    origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.5, 0.5, -0.5, 0.5),
    polarization="s", device=device)

# ---- Trace ----
tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

print(f"positions: {out['positions'].shape}")
print(f"energy_conserved: {out['energy_conserved']}")
```

### Warp tracer (fastest)

```python
from georay import WarpTracer

tracer = WarpTracer(verts, faces, face_mats, materials, device)
out = tracer.trace(rays, max_steps=20, surface_model="fresnel")
```

`WarpTracer` uses the same `trace()` signature as `AdvancedRayTracerPBRTGPU`.

### Ray from inside a medium (new)

```python
rays_inside = generate_rays(
    source_type="grid", num_rays=256,
    origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.3, 0.3, -0.3, 0.3),
    polarization="s",
    medium_id=1,          # start inside glass (material 1)
    device=device)

out_inside = tracer.trace(rays_inside, max_steps=10)
```

Pass `medium_id=<int>` to `generate_rays()` to start rays from inside any material. This works with both tracers.

---

## Examples

### 1. Wavelength-dependent glass

```python
materials = {
    0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {
        "n": {
            "wavelengths_nm": [400.0, 500.0, 600.0, 700.0],
            "values": [1.55, 1.53, 1.51, 1.49],  # chromatic dispersion
        },
        "reflectivity": {
            "wavelengths_nm": [400.0, 700.0],
            "values": [0.10, 0.05],
        },
        "diffuse": 0.0,
        "absorptivity": {
            "wavelengths_nm": [400.0, 700.0],
            "values": [0.05, 0.02],
        },
    },
}

tracer = WarpTracer(verts, faces, face_mats, materials, device)
out = tracer.trace(rays, max_steps=20, surface_model="fresnel")
```

### 2. 2D LUT (wavelength × incidence angle)

```python
materials = {
    1: {
        "n": {
            "wavelengths_nm": [400.0, 500.0, 600.0, 700.0],
            "cos_theta": [1.0, 0.707, 0.5, 0.0],   # cos(0°), cos(45°), cos(60°), cos(90°)
            "values": [
                [1.55, 1.56, 1.58, 1.60],
                [1.53, 1.54, 1.56, 1.58],
                [1.51, 1.52, 1.54, 1.56],
                [1.49, 1.50, 1.52, 1.54],
            ],
        },
        "reflectivity": 1.0,
        "diffuse": 0.0,
        "absorptivity": 0.0,
    },
}
```

### 3. Parabolic GRIN lens

```python
materials = {
    0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {
        "n": 1.5,
        "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
        "grin_field": {"type": "parabolic", "n0": 1.5, "A": 0.1},
    },
}

tracer = WarpTracer(verts, faces, face_mats, materials, device)
out = tracer.trace(rays, max_steps=200, ds_init=0.02)
```

### 4. GRIN from a 3D grid (imported from measurement / simulation)

```python
import numpy as np

x = np.linspace(-2.0, 2.0, 32).astype(np.float32)
y = np.linspace(-2.0, 2.0, 32).astype(np.float32)
z = np.linspace(-0.5, 0.5, 8).astype(np.float32)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
n_values = 1.5 - 0.15 * (X**2 + Y**2)   # shape (32, 32, 8)

materials = {
    1: {
        "n": 1.5,
        "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
        "grin_field": {
            "type": "grid",
            "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
            "values": n_values.tolist(),
        },
    },
}
```

### 5. GRIN + wavelength simultaneously (4D grid)

```python
Nx, Ny, Nz, Nw = 16, 16, 8, 4
x = np.linspace(-2.0, 2.0, Nx).astype(np.float32)
y = np.linspace(-2.0, 2.0, Ny).astype(np.float32)
z = np.linspace(-0.5, 0.5, Nz).astype(np.float32)
wl = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32)

X, Y, Z, W = np.meshgrid(x, y, z, wl, indexing="ij")
base_wl = 1.55 - 0.0001 * (W - 400.0)    # dispersion
n_4d = base_wl - 0.15 * (X**2 + Y**2)    # shape (Nx, Ny, Nz, Nw)

materials = {
    1: {
        "n": 1.5,
        "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
        "grin_field": {
            "type": "grid_4d",
            "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
            "wavelengths_nm": wl.tolist(),
            "values": n_4d.tolist(),
        },
    },
}
```

### 6. Custom GRIN via `@wp.func` (zero overhead)

```python
import warp as wp

@wp.func
def my_lens_grin(x: wp.float32, y: wp.float32, z: wp.float32):
    # Any smooth analytic expression works
    return 1.5 - 0.15 * (x*x + y*y) - 0.05 * wp.sin(z)

materials = {
    1: {
        "n": 1.5,
        "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
        "grin_field": {"type": "callable"},
    },
}

tracer = WarpTracer(verts, faces, face_mats, materials, device,
                    grin_callable=my_lens_grin)
```

### 7. STEP multi-body assembly

```python
from georay.stepmesh_adapter import glue_to_georay

# 1. Run stepmesh.glue() on three STEP files
from stepmesh import glue
result = glue(step_paths=["cube.step", "cone.step", "cylinder.step"])
vertices, faces, mapping_table = result[0], result[1], result[2]

# 2. Map volume IDs to material IDs
vol_to_material = {1: 1, 2: 2, 3: 3}

# 3. Convert to GeoRay tensors
verts_t, faces_t, face_mats_t = glue_to_georay(
    vertices, faces, mapping_table,
    vol_to_material=vol_to_material,
    outside_material_id=0,
    device=device,
)

# 4. Trace with WarpTracer (fastest)
tracer = WarpTracer(verts_t, faces_t, face_mats_t, materials, device)
out = tracer.trace(rays, max_steps=30, surface_model="fresnel")
```

See `examples/demo_stepmesh.py`, `examples/demo_stepmesh2.py`, and `examples/demo_stepmesh3.py` for complete working examples.

---

## Performance Benchmark

Environment: Google Colab, Tesla T4 GPU, 5,132-face scene, 20-step trace.

| N_rays | PyTorch BVH (ms) | Warp (ms) | Speedup |
|--------|-----------------|-----------|---------|
| 1,024  | 5,029 | 4.4   | **1140×** |
| 4,096  | 8,041 | 5.5   | **1454×** |
| 16,384 | 11,690 | 6.3  | **1856×** |
| 65,536 | 15,184 | 16.3 | **930×**  |

**Why so fast?** `WarpTracer` compiles the entire trace loop (`max_steps` iterations plus per-step BVH traversal, surface interaction, and GRIN RK4 stepping) into a **single CUDA kernel**. There is no Python-level loop, no `.any()` / `.item()` CPU-GPU synchronization, and no per-step kernel launch overhead.

---

## Feature Matrix

| Feature | AdvancedRayTracerPBRTGPU | WarpTracer |
|---------|:-:|:-:|
| Fresnel (s/p weighting) | ✅ | ✅ |
| Total internal reflection | ✅ | ✅ |
| Stokes polarization tracking | ✅ | ✅ |
| Diffuse reflection | ✅ | ✅ |
| Absorption | ✅ | ✅ |
| Monte Carlo energy weight | ✅ | ✅ |
| `surface_model` (fresnel / custom / coated) | ✅ | ✅ |
| Multi-step history | ✅ | ✅ |
| GRIN constant | ✅ | ✅ |
| GRIN parabolic | ✅ | ✅ |
| GRIN callable (`@wp.func`) | ✅ | ✅ |
| GRIN 3D grid | ✅ | ✅ |
| GRIN 4D grid (with wavelength) | ✅ | ✅ |
| Wavelength-dependent `n(λ)` | ✅ | ✅ |
| Wavelength-dependent `r(λ)`, `d(λ)`, `a(λ)` | ✅ | ✅ |
| 2D LUT `n(λ, cos θ)` | ✅ | ✅ |
| Ray from inside medium (`medium_id`) | ✅ | ✅ |
| CPU-only environments | ✅ | ❌ |

---

## Physics Validation

**209 unit tests** cover:

**Classical optics**
- Snell's law (rel. error < 1e-3 vs. analytic)
- Fresnel reflectance for s and p (abs. error < 1e-5)
- TIR critical angle
- Brewster angle
- Interface energy conservation `R + T + A = 1`
- Grazing incidence at 89.5° (numerical stability)
- Spherical interface Snell and Fresnel

**Wavelength / dispersion**
- Chromatic Snell refraction
- Wavelength-dependent `n`, `r`, `d`, `a`
- 2D LUT interpolation
- Sellmeier and Cauchy equation matching

**GRIN**
- Analytic parabolic gradient (4th-order Richardson vs. analytic)
- 3D grid trilinear interpolation
- 3D grid analytic gradient
- 4D grid quadrilinear interpolation
- Parabolic lens focusing
- Spherical aberration
- RK4 step-size convergence
- Abbe number consistency

**Polarization**
- Stokes normalization and DoP
- Jones-Mueller identity
- TIR phase retardation (documented limitation)
- Circular-to-elliptical polarization
- Mueller matrix coefficients
- DoP evolution through multiple interfaces

**Monte Carlo**
- Multiple-bounce energy invariants
- Russian roulette termination
- Diffuse cosine-weighted distribution
- Branch probability normalization

**Numerical robustness**
- CPU vs GPU determinism
- `WarpTracer` vs `AdvancedRayTracerPBRTGPU` bit-identical in deterministic scenes
- `max_steps = 0` boundary
- NaN / zero-direction input handling
- Empty mesh, non-manifold, degenerate triangles

**I/O & adapters**
- NPZ round-trip
- STL export
- STEP mesh adapter (outer / internal / degenerate triangles)

**Run:**

```bash
pytest tests/ -v
```

Expected: `209 passed`.

---

## Exceptions

GeoRay provides a hierarchy of custom exceptions, all inheriting from `GeoRayError`:

| Exception | Base class | Raised when |
|-----------|-----------|-------------|
| `GeoRayError` | `Exception` | Base class for all GeoRay exceptions |
| `ConfigurationError` | `GeoRayError`, `ValueError` | Invalid configuration (materials, scene, parameters) |
| `InvalidMaterialError` | `ConfigurationError` | Material config invalid (`n <= 0`, reflectivity out of range) |
| `InvalidRayError` | `GeoRayError`, `ValueError` | Ray data invalid (missing fields, shape mismatch) |
| `GeometryError` | `GeoRayError`, `ValueError` | Geometry invalid (empty mesh, degenerate triangles) |
| `BackendError` | `GeoRayError`, `RuntimeError` | Backend error (Warp not installed, CUDA unavailable) |

**Example:**

```python
from georay import InvalidMaterialError, InvalidRayError

try:
    bad_materials = {0: {"n": -1.5}}   # n < 0
    WarpTracer(verts, faces, fm, bad_materials, device)
except InvalidMaterialError as e:
    print(f"Invalid material: {e}")

try:
    bad_rays = {"origins": torch.zeros((4, 3), device=device)}   # missing fields
    tracer.trace(bad_rays)
except InvalidRayError as e:
    print(f"Invalid ray dict: {e}")
```

---

## API Overview

### Tracers

Both tracers expose a compatible `trace()` method:

```python
out = tracer.trace(
    ray_dict,
    max_steps=50,
    ds_init=0.02,
    ds_min=1e-3,
    ds_max=0.1,
    rk_tol=1e-4,
    fresnel=True,               # legacy flag
    surface_model=None,         # "fresnel" | "custom" | "coated"
    seed=42,                    # WarpTracer only
    rr_threshold=0.0,           # Russian roulette threshold
)
```

Returns a dict with:

| Key | Shape | Description |
|-----|-------|-------------|
| `positions` | `[N, S+1, 3]` | Ray position per step |
| `stokes` | `[N, S+1, 4]` | Stokes vector `(S0, S1, S2, S3)` |
| `active` | `[N, S+1]` | Whether the ray is still alive |
| `energy` | `[N, S+1]` | Monte Carlo path weight |
| `reflectivity` | `[N, S+1, 1]` | Per-step `R` (theoretical probability) |
| `transmissivity` | `[N, S+1, 1]` | Per-step `T` |
| `absorptivity` | `[N, S+1, 1]` | Per-step `A` |
| `n1`, `n2` | `[N, S+1, 1]` | Incident / transmitted refractive indices |
| `dop` | `[N, S+1, 1]` | Degree of polarization |
| `energy_conserved` | `bool` | `True` if invariants hold |

### Light sources

```python
generate_rays(
    source_type="grid",         # "grid" | "random" | "point_source" |
                                # "area_source" | "mesh_source"
    num_rays=1024,
    origin=(0.0, 0.0, -3.0),
    direction=(0.0, 0.0, 1.0),
    bounds=(-1, 1, -1, 1),      # OFFSET relative to origin
    polarization="s",           # "s" | "p" | "random" | angle in degrees
    wavelength_nm=550.0,
    medium_id=0,                # initial medium ID (0 = air)
    device=torch.device("cuda"),
)
```

### Material configuration

```python
materials = {
    <mat_id>: {
        "n": <spec>,                 # refractive index
        "reflectivity": <spec>,      # custom specular reflectance
        "diffuse": <spec>,           # diffuse fraction
        "absorptivity": <spec>,      # absorption fraction
        "grin_field": <grin_spec>,   # optional; presence enables GRIN
    },
    ...
}
```

**`<spec>`** (for `n`, `reflectivity`, `diffuse`, `absorptivity`):

| Form | Example |
|------|---------|
| Scalar | `1.5` |
| 1D wavelength LUT | `{"wavelengths_nm": [...], "values": [...]}` |
| 2D wavelength × cos LUT | `{"wavelengths_nm": [...], "cos_theta": [...], "values": [[...]]}` |

**`<grin_spec>`**:

| Type | Example |
|------|---------|
| Parabolic | `{"type": "parabolic", "n0": 1.5, "A": 0.1}` |
| 3D grid | `{"type": "grid", "x": [...], "y": [...], "z": [...], "values": [[[...]]]}` |
| 4D grid | `{"type": "grid_4d", "x": [...], "y": [...], "z": [...], "wavelengths_nm": [...], "values": [[[[...]]]]}` |
| Callable | `{"type": "callable"}` + `grin_callable=my_wp_func` |

### I/O

```python
from georay import save_trace_output_to_npz, export_ray_history_to_stl

save_trace_output_to_npz(out, "trace.npz")
export_ray_history_to_stl(out, "rays.stl", radius=0.02, sides=6)
```

### STEP mesh adapter

```python
from georay.stepmesh_adapter import glue_to_georay

verts, faces, face_mats = glue_to_georay(
    vertices, faces, mapping_table,
    vol_to_material={1: 10, 2: 20},
    outside_material_id=0,
    device=device,
)
```

---

## Project Layout

```
georay/
├── georay/
│   ├── __init__.py             # Public API
│   ├── tracer.py               # PyTorch BVH tracer
│   ├── warp_tracer.py          # Warp kernel tracer (fastest)
│   ├── bvh.py                  # Pure PyTorch BVH
│   ├── grin.py                 # GRIN field definitions
│   ├── integrator.py           # RK4 stepping
│   ├── interpolation.py        # GPU interpolation utilities
│   ├── polarization.py         # Jones-Mueller polarization utilities
│   ├── sources.py              # Light source generation
│   ├── io.py                   # NPZ / STL export
│   ├── stepmesh_adapter.py     # STEP → GeoRay adapter
│   ├── thin_film.py            # Thin-film interference (transfer matrix)
│   ├── grating.py              # Diffraction grating equations
│   ├── nonlinear.py            # Kerr nonlinear optics
│   └── exceptions.py           # Custom exception classes
├── tests/                      # 209 unit tests
├── examples/                   # Example scripts and STEP files
│   ├── colab_quickstart.py     # Colab quickstart
│   ├── demo_cube.py            # Basic cube tracing
│   ├── demo_grin.py            # GRIN lens
│   ├── demo_polarization.py    # Polarization tracking
│   ├── demo_stepmesh.py        # STEP multi-body assembly
│   ├── demo_stepmesh2.py       # STEP with custom ray layout
│   ├── demo_stepmesh3.py       # Watertight control experiment
│   ├── quick_check_stepfiles.py # Quick STEP validation
│   ├── cube.step               # Example STEP files
│   ├── cone.step
│   └── cylinder.step
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── CITATION.cff
```

---

## Known Limitations

- **Rays cannot start inside a medium** — *Resolved in v1.2* via `medium_id` parameter.
- **Complex refractive index** (metals, absorbing dielectrics) not supported.
- **TIR phase retardation** is not tracked (real-valued Fresnel coefficients only).
- **Thin-film interference, anisotropy, volumetric scattering** not implemented.
- **`WarpTracer`** requires a CUDA GPU.

---

## Citation
If you use GeoRay in your research, please cite:

**Paper (Preprint, Zenodo):**

> G. Wang, "GeoRay: A GPU-Accelerated Ray Tracer for Gradient-Index and Polarization Optics," *Zenodo*, 2026. DOI: [10.5281/zenodo.23284911](https://doi.org/10.5281/zenodo.23284911)

**BibTeX:**

```bibtex
@misc{wang2026georay,
  title     = {GeoRay: A GPU-Accelerated Ray Tracer for Gradient-Index and Polarization Optics},
  author    = {Wang, Gang},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23284911},
  url       = {https://doi.org/10.5281/zenodo.23284911}
}
```

**Software (GitHub):**

> G. Wang, "GeoRay: A GPU-Accelerated Ray Tracer for Gradient-Index and Polarization Optics," GitHub repository, 2026. URL: 

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Warp](https://github.com/NVIDIA/warp) by NVIDIA — GPU kernel programming framework
- [trimesh](https://github.com/mikedh/trimesh) — mesh loading and processing
- [PyTorch](https://pytorch.org) — tensor computation
- [stepmesh](https://github.com/gwangcode/stepmesh) — STEP file remeshing