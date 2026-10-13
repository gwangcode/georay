# GeoRay

[![DOI](https://zenodo.org/badge/1389842626.svg)](https://doi.org/10.5281/zenodo.22983753)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Tests](https://github.com/gwangcode/georay/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/gwangcode/georay/actions/workflows/test.yml)

A GPU-Accelerated Ray Tracer for GRIN and Polarization Optics.

GeoRay is a research- and engineering-oriented ray tracing library supporting:

Fresnel reflection/refraction, TIR, diffuse reflection, and absorption on triangle meshes

Full Stokes-vector polarization tracking via the Jones-Mueller formalism

Adaptive RK4 stepping inside gradient-index (GRIN) media

Warp GPU kernel implementation: 1000×+ speedup over the pure PyTorch BVH

## Installation
Basic install (PyTorch BVH fallback)
bash
pip install git+https://github.com/gwangcode/georay
With Warp GPU acceleration (recommended)
bash
pip install "georay[warp] @ git+https://github.com/gwangcode/georay"
Warp requires a CUDA GPU (Compute Capability ≥ 5.0). Colab T4 is fully supported.

## Quick Start
PyTorch tracer (flexible; supports GRIN and wavelength-dependent materials)
python
import torch, trimesh
from georay import AdvancedRayTracerPBRTGPU, generate_rays

device = torch.device("cuda")

## Scene
cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
face_mats[:, 0] = 0
face_mats[:, 1] = 1
materials = {
    0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
}

## Rays
rays = generate_rays(source_type="grid", num_rays=1024,
                     origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
                     bounds=(-0.5, 0.5, -0.5, 0.5),
                     polarization="s", device=device)

## Trace
tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
out = tracer.trace(rays, max_steps=20, surface_model="fresnel")
Warp tracer (fastest; scalar materials + parabolic GRIN only)
python
from georay import WarpTracer

tracer = WarpTracer(verts, faces, face_mats, materials, device)
out = tracer.trace(rays, max_steps=20, surface_model="fresnel")
GRIN parabolic lens (Warp version)
python
materials = {
    0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0,
        "grin_field": {"type": "parabolic", "n0": 1.5, "A": 0.1}},
}
tracer = WarpTracer(verts, faces, face_mats, materials, device)
out = tracer.trace(rays, max_steps=200, ds_init=0.02)
Performance Benchmark
Environment: Colab T4 GPU, 5,132-face scene, 20-step trace.

N_rays	PyTorch BVH (ms)	Warp (ms)	Speedup
1,024	5,029	4.4	1140×
4,096	8,041	5.5	1454×
16,384	11,690	6.3	1856×
65,536	15,184	16.3	930×
Key insight: WarpTracer compiles the entire trace loop (all max_steps iterations) into a single CUDA kernel, eliminating the Python-level loop and all CPU-GPU synchronization.

Feature Matrix
Feature	AdvancedRayTracerPBRTGPU	WarpTracer
Fresnel (s/p polarization-weighted)	✅	✅
Total internal reflection (TIR)	✅	✅
Stokes polarization tracking	✅	✅
Diffuse reflection	✅	✅
Absorption	✅	✅
Monte Carlo energy weight	✅	✅
surface_model (fresnel/custom/coated)	✅	✅
Multi-step history	✅	✅
GRIN constant field	✅	✅
GRIN parabolic	✅	✅
GRIN callable / dict-grid / wavelength-dependent	✅	❌
Wavelength-dependent n(λ)	✅	❌
Ray starting inside a medium	❌	❌
Complex refractive index	❌	❌
Thin-film interference	❌	❌
Anisotropic media	❌	❌
CPU-only environments	✅	❌

## API Overview
### Main classes
AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
PyTorch BVH tracer. Full feature set, best for flexible research scenarios.

WarpTracer(verts, faces, face_mats, materials, device)
Warp kernel tracer. Best performance, best for large-batch, multi-step simulations.

Both expose a compatible trace() API:

python
out = tracer.trace(
    ray_dict,
    max_steps=50,
    ds_init=0.02,
    fresnel=True,              # legacy flag
    surface_model=None,        # "fresnel" | "custom" | "coated"
)
Light sources
generate_rays(...) supports grid, random, point_source, area_source, and mesh_source.

I/O
save_trace_output_to_npz(out, path)

export_ray_history_to_stl(out, path)

STEP mesh adapter
georay.stepmesh_adapter.glue_to_georay(...) converts stepmesh.glue() output into GeoRay inputs.

Physics Validation
44 unit tests cover:

Snell's law (angle vs. analytical, error < 1e-3)

Fresnel reflectance (s/p, error < 1e-5)

TIR critical angle

Brewster angle

Interface energy conservation: R + T + A = 1

Dispersion (Snell angle vs. wavelength)

GRIN gradient (4th-order Richardson extrapolation vs. analytical)

Multiple-bounce energy invariants

Diffuse cosine-weighted distribution (mean cos θ ≈ 2/3)

Stokes normalization and DoP

Russian roulette invariants

Grazing incidence at 89.5° (numerical stability)

BVH hit / miss

NPZ / STL I/O round-trips

STEP mesh degenerate-triangle handling

Run:

bash
pytest tests/ -v
Expected: 44 passed.

Colab One-Click Test
python
# Install
!pip install -q "georay[warp] @ git+https://github.com/gwangcode/georay"

# Verify
import georay
print(f"version: {georay.__version__}")
print(f"WarpTracer: {'available' if georay.WarpTracer else 'NOT available'}")

# Quick benchmark
import torch, trimesh, time
from georay import AdvancedRayTracerPBRTGPU, WarpTracer, generate_rays

device = torch.device("cuda")
cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
fm[:, 0] = 0; fm[:, 1] = 1
mats = {0: {"n": 1.0}, 1: {"n": 1.5}}

rays = generate_rays(source_type="grid", num_rays=16384,
                     origin=(0, 0, -3), direction=(0, 0, 1),
                     bounds=(-0.5, 0.5, -0.5, 0.5),
                     polarization="s", device=device)

for name, cls in [("PyTorch", AdvancedRayTracerPBRTGPU), ("Warp", WarpTracer)]:
    tracer = cls(verts, faces, fm, mats, device)
    tracer.trace(rays, max_steps=2)      # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tracer.trace(rays, max_steps=20)
    torch.cuda.synchronize()
    print(f"{name:8s}: {(time.perf_counter() - t0) * 1000:.2f} ms")
Known Limitations
WarpTracer
Scalar material properties only (no wavelength-dependent or dict-grid GRIN)

Parabolic GRIN only (use AdvancedRayTracerPBRTGPU for other GRIN types)

Requires a CUDA GPU (T4 or better)

General
Rays cannot start inside a medium (API-level gap)

Complex refractive index not supported (metals, absorbing media)

TIR phase retardation not implemented (real-valued Fresnel coefficients only)

No thin-film interference, anisotropy, or volumetric scattering

Project Layout
text
georay/
├── georay/
│   ├── __init__.py             # Public API
│   ├── tracer.py               # PyTorch BVH tracer
│   ├── warp_tracer.py          # Warp kernel tracer
│   ├── bvh.py                  # Pure PyTorch BVH
│   ├── grin.py                 # GRIN field definitions
│   ├── integrator.py           # RK4 stepping
│   ├── interpolation.py        # GPU interpolation
│   ├── polarization.py         # Jones-Mueller polarization utilities
│   ├── sources.py              # Light source generation
│   ├── io.py                   # NPZ / STL export
│   └── stepmesh_adapter.py     # STEP → GeoRay adapter
├── tests/                      # 44 unit tests
├── examples/                   # Example scripts
├── pyproject.toml
├── requirements.txt
└── README.md
Citation
If you use GeoRay in your research, please cite:

bibtex
@software{georay2026,
  author = {Wang, Gang},
  title = {GeoRay: A GPU-Accelerated Ray Tracer for GRIN and Polarization Optics},
  year = {2026},
  url = {https://github.com/gwangcode/georay}
}
License
MIT License. See LICENSE for details.

## Citation
If you use GeoRay in your research, please cite our paper:

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

> G. Wang, "GeoRay: A GPU-Accelerated Ray Tracer for Gradient-Index and Polarization Optics," GitHub repository, 2026. URL: [https://github.com/gwangcode/georay](https://github.com/gwangcode/georay)

## License
MIT License. See LICENSE for details.

## Contributing
Issues and pull requests are welcome on the GitHub repository.
