# GeoRay Tutorial

**Version**: 1.1.0  
**Audience**: Researchers and students in computational optics, photonics, and physics

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
5. [Light Sources](#5-light-sources)
6. [Material Configuration](#6-material-configuration)
7. [Using the Tracers](#7-using-the-tracers)
8. [GRIN Field Configuration](#8-grin-field-configuration)
9. [Polarization Tracking](#9-polarization-tracking)
10. [Input and Output](#10-input-and-output)
11. [STEP CAD Integration](#11-step-cad-integration)
12. [Exception Handling](#12-exception-handling)
13. [Performance Optimization](#13-performance-optimization)
14. [Complete Examples](#14-complete-examples)
15. [FAQ](#15-faq)

---

## 1. Introduction

GeoRay is a PyTorch-based, GPU-accelerated ray tracing library focused on **gradient-index (GRIN) media** and **polarization optics**. It provides two tracer backends:

| Tracer | Characteristics | Best for |
|--------|-----------------|----------|
| `AdvancedRayTracerPBRTGPU` | Pure PyTorch BVH, CPU/GPU | Flexible research, custom callable fields |
| `WarpTracer` | Full Warp kernel, single CUDA kernel launch | Large batches, high performance |

**Key features**:
- Fresnel reflection/refraction (s/p polarization-weighted)
- Total internal reflection (TIR)
- Stokes polarization tracking (Jones-Mueller formalism)
- Diffuse reflection (cosine-weighted hemisphere sampling)
- Absorption (Monte Carlo energy weighting)
- Multiple bounces and Russian roulette
- Wavelength-dependent materials (1D/2D LUT)
- GRIN media (parabolic, 3D grid, 4D grid, callable function)
- STEP CAD multi-body assembly integration

---

## 2. Installation

### 2.1 Basic install (PyTorch BVH only)

```bash
pip install git+https://github.com/gwangcode/georay
```

### 2.2 With Warp GPU acceleration (recommended)

```bash
pip install "georay[warp] @ git+https://github.com/gwangcode/georay"
```

### 2.3 Full install (Warp + STEP + dev tools)

```bash
# Option 1: install directly from GitHub
pip install "georay[all] @ git+https://github.com/gwangcode/georay"

# Option 2: clone and install locally
git clone https://github.com/gwangcode/georay.git
cd georay
pip install -e ".[all]"
```

### 2.4 System requirements

| Component | Requirement |
|-----------|-------------|
| Python | ≥ 3.9 (3.10+ recommended) |
| PyTorch | ≥ 2.0 |
| trimesh | ≥ 3.20 |
| Warp (optional) | ≥ 1.0, CUDA GPU required (Compute Capability ≥ 5.0) |
| stepmesh (optional) | Install from GitHub |

**Verify the install**:

```python
import georay
print(f"GeoRay version: {georay.__version__}")
print(f"WarpTracer: {'available' if georay.WarpTracer else 'NOT available'}")
```

---

## 3. Quick Start

### 3.1 Minimal example

```python
import torch
import trimesh
from georay import AdvancedRayTracerPBRTGPU, generate_rays

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scene: 2x2x2 glass cube
cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(cube.faces, dtype=torch.long, device=device)

# Face materials: [:, 0] = front (outward normal), [:, 1] = back
face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
face_mats[:, 0] = 0   # front -> air
face_mats[:, 1] = 1   # back  -> glass

# Material definitions
materials = {
    0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
}

# Generate 1024 parallel rays
rays = generate_rays(
    source_type="grid", num_rays=1024,
    origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.5, 0.5, -0.5, 0.5),
    polarization="s", device=device)

# Trace
tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

# Inspect results
print(f"positions shape: {out['positions'].shape}")   # [N, S+1, 3]
print(f"energy_conserved: {out['energy_conserved']}")
```

### 3.2 Warp-accelerated version (same scene)

```python
from georay import WarpTracer

tracer = WarpTracer(verts, faces, face_mats, materials, device)
out = tracer.trace(rays, max_steps=20, surface_model="fresnel", seed=42)
```

`WarpTracer` shares the same `trace()` signature and output format as `AdvancedRayTracerPBRTGPU`.

---

## 4. Core Concepts

### 4.1 Scene representation

GeoRay represents geometry with triangle meshes:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `vertices` | `[V, 3]` | Vertex coordinates (float32) |
| `faces` | `[F, 3]` | Triangle vertex indices (long) |
| `face_mats` | `[F, 2]` | Two-sided material IDs per face (long) |

**`face_mats` semantics**:
- `face_mats[i, 0]`: material ID on the **front side** of face i (along the normal)
- `face_mats[i, 1]`: material ID on the **back side** of face i

### 4.2 Output dictionary

`trace()` returns a dictionary with:

| Key | Shape | Description |
|-----|-------|-------------|
| `positions` | `[N, S+1, 3]` | Ray position per step |
| `stokes` | `[N, S+1, 4]` | Stokes vector `(S0, S1, S2, S3)` |
| `active` | `[N, S+1]` | Whether the ray is still alive |
| `energy` | `[N, S+1]` | Monte Carlo path weight |
| `reflectivity` | `[N, S+1, 1]` | Per-step reflectance (theoretical probability) |
| `transmissivity` | `[N, S+1, 1]` | Per-step transmittance |
| `absorptivity` | `[N, S+1, 1]` | Per-step absorptance |
| `n1`, `n2` | `[N, S+1, 1]` | Incident/outgoing refractive indices |
| `dop` | `[N, S+1, 1]` | Degree of polarization |
| `energy_conserved` | `bool` | Whether invariants hold |

### 4.3 Wavelength and polarization

- Each ray carries a **wavelength** `wavelengths[N, 1]` (nm)
- **Stokes vector** `[N, 4]`: `S0` = total intensity, `S1` = horizontal/vertical linear, `S2` = ±45° linear, `S3` = circular
- Reference frames `s_ref` and `p_ref` define the local polarization basis

---

## 5. Light Sources

`generate_rays()` is the unified light-source interface, supporting 5 source types.

### 5.1 Full signature

```python
generate_rays(
    source_type="grid",          # source type
    num_rays=100,                # number of rays
    origin=(0, 0, 0),            # source center
    direction=(0, 0, 1),         # main propagation direction
    radius=1.0,                  # disk radius for "random"
    angle_deg=10.0,              # cone half-angle for "point_source" (deg)
    bounds=(-1, 1, -1, 1),       # offset range for "grid"/"area_source"
    mesh=None,                   # trimesh object for "mesh_source"
    polarization="random",       # "s" | "p" | "random" | angle (deg)
    wavelength_nm=550.0,         # wavelength (nm)
    medium_id=0,                 # initial medium ID (0 = air)
    device=torch.device("cpu"),
)
```

> **Note**: `bounds` is an **offset relative to `origin`**, not absolute coordinates. For example, `origin=(0,0,-3)` + `bounds=(-0.5, 0.5, -0.5, 0.5)` generates rays with x ∈ [-0.5, 0.5] and y ∈ [-0.5, 0.5].

### 5.2 Grid source

```python
rays = generate_rays(
    source_type="grid", num_rays=1024,
    origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.5, 0.5, -0.5, 0.5),
    polarization="s", device=device)
```

- `num_rays` is rounded down to the nearest perfect square (e.g., 1000 → 961 = 31²)
- Produces a uniform grid of rays

### 5.3 Random source

```python
rays = generate_rays(
    source_type="random", num_rays=500, radius=2.0,
    origin=(1.0, 1.0, 0.0), direction=(0.0, 0.0, 1.0),
    device=device)
```

Uniformly samples within a disk perpendicular to `direction`.

### 5.4 Point source (cone)

```python
rays = generate_rays(
    source_type="point_source", num_rays=500, angle_deg=30.0,
    origin=(0.5, -0.5, 0.0), direction=(0.0, 0.0, 1.0),
    device=device)
```

All rays start from the same point, with directions uniformly distributed within the cone half-angle `angle_deg`.

### 5.5 Area source

```python
rays = generate_rays(
    source_type="area_source", num_rays=500,
    bounds=(-1.0, 1.0, -2.0, 2.0),
    origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
    device=device)
```

Uniformly samples a rectangular area, with cosine-weighted hemisphere directions.

### 5.6 Mesh source

```python
mesh = trimesh.creation.icosphere()
rays = generate_rays(
    source_type="mesh_source", num_rays=500, mesh=mesh,
    device=device)
```

Samples the mesh surface with area weighting, with cosine-weighted directions along the normal hemisphere.

### 5.7 Rays starting from inside a medium

```python
rays_inside = generate_rays(
    source_type="grid", num_rays=256,
    origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.3, 0.3, -0.3, 0.3),
    medium_id=1,          # start inside material 1 (glass)
    device=device)
```

`medium_id` can be an integer (all rays identical) or a list (per-ray different).

---

## 6. Material Configuration

### 6.1 Basic structure

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

### 6.2 Property specifications (`<spec>`)

**Scalar constant**:
```python
"n": 1.5
"reflectivity": 0.8
```

**1D wavelength LUT**:
```python
"n": {
    "wavelengths_nm": [400.0, 500.0, 600.0, 700.0],
    "values": [1.55, 1.53, 1.51, 1.49],
}
```

**2D wavelength × angle LUT**:
```python
"n": {
    "wavelengths_nm": [400.0, 500.0, 600.0, 700.0],
    "cos_theta": [1.0, 0.707, 0.5, 0.0],
    "values": [
        [1.55, 1.56, 1.58, 1.60],
        [1.53, 1.54, 1.56, 1.58],
        [1.51, 1.52, 1.54, 1.56],
        [1.49, 1.50, 1.52, 1.54],
    ],
}
```

### 6.3 Surface models

| Model | Description |
|-------|-------------|
| `"fresnel"` | Pure Fresnel reflectance |
| `"custom"` | Pure user-supplied reflectance |
| `"coated"` | `R_eff × custom_refl` (coated modulation) |

```python
out = tracer.trace(rays, surface_model="fresnel")
```

---

## 7. Using the Tracers

### 7.1 Full signature

```python
out = tracer.trace(
    ray_dict,
    max_steps=50,          # maximum number of steps
    ds_init=0.02,          # initial step size
    ds_min=1e-3,           # minimum step size
    ds_max=0.1,            # maximum step size
    rk_tol=1e-4,           # RK4 tolerance
    fresnel=True,          # legacy flag
    surface_model=None,    # "fresnel" | "custom" | "coated"
    seed=42,               # WarpTracer only
    rr_threshold=0.0,      # Russian roulette threshold
)
```

### 7.2 Parameter descriptions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 50 | Maximum number of trace steps |
| `ds_init` | 0.02 | Initial step size (GRIN stepping) |
| `ds_min` / `ds_max` | 1e-3 / 0.1 | Step size bounds |
| `rk_tol` | 1e-4 | RK4 local error tolerance |
| `surface_model` | `None` | See section 6.3 |
| `rr_threshold` | 0.0 | Enable Russian roulette when > 0 |

### 7.3 PyTorch vs Warp tracer

```python
# PyTorch tracer: supports custom Python callable fields
from georay import AdvancedRayTracerPBRTGPU
tracer_pt = AdvancedRayTracerPBRTGPU(verts, faces, fm, mats, device)

# Warp tracer: faster, requires CUDA
from georay import WarpTracer
tracer_wp = WarpTracer(verts, faces, fm, mats, device)

# Both produce the same output format
out_pt = tracer_pt.trace(rays, max_steps=20)
out_wp = tracer_wp.trace(rays, max_steps=20, seed=42)
```

---

## 8. GRIN Field Configuration

### 8.1 Parabolic field (analytic)

```python
materials = {
    1: {
        "n": 1.5,
        "grin_field": {"type": "parabolic", "n0": 1.5, "A": 0.1},
    },
}
```

`n(r) = n0 - A·(x² + y²)`, with built-in RK4 and analytic gradient.

### 8.2 3D grid (imported from measurement or simulation)

```python
import numpy as np

x = np.linspace(-2.0, 2.0, 32).astype(np.float32)
y = np.linspace(-2.0, 2.0, 32).astype(np.float32)
z = np.linspace(-0.5, 0.5, 8).astype(np.float32)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
n_values = 1.5 - 0.15 * (X**2 + Y**2)

materials = {
    1: {
        "n": 1.5,
        "grin_field": {
            "type": "grid",
            "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
            "values": n_values.tolist(),
        },
    },
}
```

Uses trilinear interpolation plus analytic gradient.

### 8.3 4D grid (with wavelength)

```python
Nx, Ny, Nz, Nw = 16, 16, 8, 4
x = np.linspace(-2.0, 2.0, Nx).astype(np.float32)
y = np.linspace(-2.0, 2.0, Ny).astype(np.float32)
z = np.linspace(-0.5, 0.5, Nz).astype(np.float32)
wl = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32)

X, Y, Z, W = np.meshgrid(x, y, z, wl, indexing="ij")
base_wl = 1.55 - 0.0001 * (W - 400.0)
n_4d = base_wl - 0.15 * (X**2 + Y**2)

materials = {
    1: {
        "n": 1.5,
        "grin_field": {
            "type": "grid_4d",
            "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
            "wavelengths_nm": wl.tolist(),
            "values": n_4d.tolist(),
        },
    },
}
```

### 8.4 Custom `@wp.func` (zero overhead)

```python
import warp as wp

@wp.func
def my_lens_grin(x: wp.float32, y: wp.float32, z: wp.float32):
    return 1.5 - 0.15 * (x*x + y*y) - 0.05 * wp.sin(z)

materials = {
    1: {
        "n": 1.5,
        "grin_field": {"type": "callable"},
    },
}

tracer = WarpTracer(verts, faces, fm, materials, device,
                    grin_callable=my_lens_grin)
```

Compile-time inlined, **zero runtime overhead**.

---

## 9. Polarization Tracking

### 9.1 Stokes vector

| Component | Physical meaning |
|-----------|------------------|
| `S0` | Total intensity (normalized to 1) |
| `S1` | Horizontal (+) / vertical (−) linear polarization |
| `S2` | +45° (−) / −45° (+) linear polarization |
| `S3` | Right (+) / left (−) circular polarization |

### 9.2 Polarization input

```python
# s-polarization (S1 = +1)
rays = generate_rays(..., polarization="s")

# p-polarization (S1 = −1)
rays = generate_rays(..., polarization="p")

# Unpolarized (S1 = S2 = S3 = 0)
rays = generate_rays(..., polarization="random")

# Arbitrary linear angle
rays = generate_rays(..., polarization=45.0)   # +45°
```

### 9.3 Polarization evolution

```python
out = tracer.trace(rays, max_steps=20)
stokes = out["stokes"]     # [N, S+1, 4]
dop = out["dop"]           # [N, S+1, 1] degree of polarization
```

**Physical rules for DoP**:
- Pure polarized light (DoP=1) remains DoP=1 after specular reflection
- DoP collapses to 0 after diffuse reflection (fully depolarized)
- Partially polarized light evolves via the Mueller matrix at each interface

---

## 10. Input and Output

### 10.1 NPZ export

```python
from georay import save_trace_output_to_npz

save_trace_output_to_npz(out, "trace.npz")

# Load
import numpy as np
data = np.load("trace.npz")
positions = data["positions"]
```

### 10.2 STL export

```python
from georay import export_ray_history_to_stl

export_ray_history_to_stl(
    out, "rays.stl",
    radius=0.02,   # ray cylinder radius
    sides=6,       # cylinder faces
)
```

The exported STL can be viewed in FreeCAD, Blender, etc.

---

## 11. STEP CAD Integration

### 11.1 Complete workflow

```python
from stepmesh import glue
from georay.stepmesh_adapter import glue_to_georay

# 1. Process STEP files
vertices, faces, mapping_table = glue(
    step_paths=["cube.step", "cone.step", "cylinder.step"],
    return_internal_surfaces=True,
)[:3]

# 2. Build volume ID -> material ID mapping
all_vols = sorted(set(mapping_table[:, 0].tolist()) |
                  set(mapping_table[:, 1].tolist()))
vol_to_material = {0: 0}
for i, v in enumerate([v for v in all_vols if v != 0]):
    vol_to_material[v] = (i % 3) + 1

# 3. Convert to GeoRay tensors
verts_t, faces_t, face_mats_t = glue_to_georay(
    vertices, faces, mapping_table,
    vol_to_material=vol_to_material,
    outside_material_id=0,
    device=device,
)

# 4. Trace
tracer = WarpTracer(verts_t, faces_t, face_mats_t, materials, device)
out = tracer.trace(rays, max_steps=30, surface_model="fresnel")
```

### 11.2 Key notes

- `return_internal_surfaces=True` **must be enabled**, otherwise internal interfaces are lost
- `face_mats` automatically handles two-sided materials; internal interfaces have different materials on each side
- Coplanar contact issues are handled automatically by stepmesh's `glue()`

---

## 12. Exception Handling

### 12.1 Exception hierarchy

| Exception | Base class | Raised when |
|-----------|------------|-------------|
| `GeoRayError` | `Exception` | Base class for all GeoRay exceptions |
| `ConfigurationError` | `GeoRayError`, `ValueError` | Invalid configuration |
| `InvalidMaterialError` | `ConfigurationError` | Material config invalid |
| `InvalidRayError` | `GeoRayError`, `ValueError` | Ray data invalid |
| `GeometryError` | `GeoRayError`, `ValueError` | Geometry data invalid |
| `BackendError` | `GeoRayError`, `RuntimeError` | Backend error |

### 12.2 Usage example

```python
from georay import InvalidMaterialError, InvalidRayError

# Invalid material -> raises at construction
try:
    WarpTracer(verts, faces, fm, {0: {"n": -1.5}}, device)
except InvalidMaterialError as e:
    print(f"Invalid material: {e}")

# Invalid ray dict -> raises at trace
try:
    tracer.trace({"origins": torch.zeros((4, 3), device=device)})
except InvalidRayError as e:
    print(f"Invalid ray dict: {e}")
```

---

## 13. Performance Optimization

### 13.1 Choosing the right backend

| Scenario | Recommended backend |
|----------|--------------------|
| Few rays (< 1000) | PyTorch (fast even on CPU) |
| Many rays (> 10,000) | Warp (GPU) |
| Custom Python callable fields | PyTorch |
| Maximum throughput | Warp |

### 13.2 Performance benchmark (T4 GPU)

| Number of rays | PyTorch (ms) | Warp (ms) | Speedup |
|----------------|-------------|-----------|---------|
| 1,024 | 5,029 | 4.4 | **1,140×** |
| 4,096 | 8,041 | 5.5 | **1,454×** |
| 16,384 | 11,690 | 6.3 | **1,856×** |
| 65,536 | 15,184 | 16.3 | **930×** |

### 13.3 Optimization tips

- **Batch rays**: Warp's advantage appears only when N ≥ 10,000
- **Warm-up**: The first Warp call compiles the kernel (~30s); subsequent calls are millisecond-scale
- **Step size**: For GRIN scenes use `ds_init=0.05, ds_max=0.2`
- **Tolerance**: Tighten `rk_tol` to `1e-5` for strongly inhomogeneous fields
- **Caching**: Warp kernel cache is stored at `~/.cache/warp/`

---

## 14. Complete Examples

### 14.1 Multi-body STEP assembly tracing

```python
import os
import numpy as np
import torch
import trimesh
from georay import WarpTracer, generate_rays, export_ray_history_to_stl
from georay.stepmesh_adapter import glue_to_georay
from stepmesh import glue

device = torch.device("cuda")

# 1. Process STEP
vertices, faces, mapping_table = glue(
    step_paths=["cube.step", "cone.step", "cylinder.step"],
    return_internal_surfaces=True,
)[:3]

# 2. Volume -> material mapping
all_vols = sorted(set(mapping_table[:, 0].tolist()) |
                  set(mapping_table[:, 1].tolist()))
vol_to_material = {0: 0}
for i, v in enumerate([v for v in all_vols if v != 0]):
    vol_to_material[v] = (i % 3) + 1

# 3. Convert
verts_t, faces_t, face_mats_t = glue_to_georay(
    vertices, faces, mapping_table,
    vol_to_material=vol_to_material,
    outside_material_id=0,
    device=device)

# 4. Materials
materials = {
    0: {"n": 1.00, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.50, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    2: {"n": 1.77, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    3: {"n": 1.33, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
}

# 5. Rays
v_np = verts_t.cpu().numpy()
bmin, bmax = v_np.min(axis=0), v_np.max(axis=0)
center = 0.5 * (bmin + bmax)
size = bmax - bmin

rays = generate_rays(
    source_type="grid", num_rays=25,
    origin=(float(center[0]), float(center[1]),
            float(bmin[2] - 0.5 * size[2])),
    direction=(0.0, 0.0, 1.0),
    bounds=(-0.4 * size[0], 0.4 * size[0],
            -0.4 * size[1], 0.4 * size[1]),
    polarization="random", device=device)

# 6. Trace
tracer = WarpTracer(verts_t, faces_t, face_mats_t, materials, device)
out = tracer.trace(rays, max_steps=30, surface_model="fresnel")

# 7. Export
export_ray_history_to_stl(
    out, "ray_tracks.stl",
    radius=0.005 * float(size.max()), sides=8)
```

### 14.2 Parabolic GRIN lens focusing

```python
import torch
import trimesh
from georay import AdvancedRayTracerPBRTGPU, generate_rays

device = torch.device("cpu")

# Scene: 1x1x2 GRIN medium box
box = trimesh.creation.box(extents=[1.0, 1.0, 2.0])
verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(box.faces, dtype=torch.long, device=device)
fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
fm[:, 0] = 0
fm[:, 1] = 1

materials = {
    0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
        "absorptivity": 0.0,
        "grin_field": {"type": "parabolic", "n0": 1.5, "A": 0.15}},
}

# Off-axis rays
rays = generate_rays(
    source_type="grid", num_rays=16,
    origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.4, 0.4, -0.4, 0.4),
    polarization="s", device=device)

tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
out = tracer.trace(rays, max_steps=200, ds_init=0.02)

# Check focusing: |x_end| < |x_start|
x_start = out["positions"][:, 0, 0].abs().mean().item()
x_end = out["positions"][:, -1, 0].abs().mean().item()
print(f"|x_start| = {x_start:.4f}, |x_end| = {x_end:.4f}")
assert x_end < x_start, "GRIN lens did not focus"
```

### 14.3 Dispersive glass prism

```python
materials = {
    1: {
        "n": {
            "wavelengths_nm": [400.0, 500.0, 600.0, 700.0],
            "values": [1.55, 1.53, 1.51, 1.49],
        },
        "reflectivity": 0.0,
        "diffuse": 0.0,
        "absorptivity": 0.0,
    },
}

# Different wavelengths
for wl in [400.0, 700.0]:
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(-2.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device, wavelength_nm=wl)
    out = tracer.trace(rays, max_steps=5, surface_model="fresnel")
    print(f"λ={wl}nm: n = {out['n2'][0, 1, 0].item():.4f}")
```

---

## 15. FAQ

### Q1: Why is the first `WarpTracer` call slow?

Warp compiles PTX/SASS code for each kernel variant on first use (~30s). Subsequent calls are millisecond-scale. Compiled results are cached at `~/.cache/warp/`.

### Q2: What if `num_rays` is not a perfect square?

Grid sources round down to the nearest perfect square. For example, `num_rays=1000` produces 961 rays (31²). Use `rays["origins"].shape[0]` for the actual count.

### Q3: Rays "vanish" from the scene?

Check the `out["active"]` array. A ray's `active` becoming `False` means it has terminated (escaped, absorbed, or reached `max_steps`).

### Q4: How to debug GRIN stepping?

`WarpTracer` does not support verbose mode. Use `AdvancedRayTracerPBRTGPU` and inspect `out["rk_error"]` and `out["ds_history"]`.

### Q5: STEP mesh is not watertight after processing?

stepmesh output may contain non-manifold edges. These do not affect ray tracing (GeoRay's BVH handles them), but may matter for other downstream applications. Check `summarize_glue_output()`.

### Q6: How to choose `surface_model`?

| Need | Recommended |
|------|-------------|
| Pure physical Fresnel | `"fresnel"` |
| User-supplied reflectance | `"custom"` |
| Coated modulation | `"coated"` |

### Q7: How does `medium_id` work?

`generate_rays(..., medium_id=1)` makes rays start inside material 1. The tracer correctly identifies `n1` (medium side) and `n2` (exterior side) at the first interface.

### Q8: How to get more performance?

1. Use `WarpTracer` with `device=torch.device("cuda")`
2. Increase batch size (N ≥ 10,000)
3. Reduce `max_steps`
4. Increase `ds_init` / `ds_max`

---

## Appendix: Module Reference

| Module | Main functionality |
|--------|-------------------|
| `georay.tracer` | `AdvancedRayTracerPBRTGPU` |
| `georay.warp_tracer` | `WarpTracer` |
| `georay.sources` | `generate_rays` |
| `georay.grin` | `BaseGRINField`, `parse_grin_input_to_field` |
| `georay.integrator` | `rk4_step_grin`, `adaptive_rk4_step_per_ray` |
| `georay.polarization` | Jones-Mueller utilities |
| `georay.io` | NPZ/STL export |
| `georay.stepmesh_adapter` | STEP mesh conversion |
| `georay.exceptions` | Custom exceptions |
| `georay.thin_film` | Thin-film interference (transfer matrix) |
| `georay.grating` | Diffraction grating equations |
| `georay.nonlinear` | Kerr nonlinear optics |

---

**More information**:
- Repository: https://github.com/gwangcode/georay
- Issues: https://github.com/gwangcode/georay/issues
- Examples: the `demo_*.py` files in `examples/`
