"""
GeoRay Colab Quickstart
=======================
Copy and paste into Google Colab cells to run.

Prerequisites: enable GPU runtime
    Colab menu -> Runtime -> Change runtime type -> T4 GPU
"""

# ==============================================================================
# Cell 1: Install
# ==============================================================================
# !pip install -q "georay[warp] @ git+https://github.com/gwangcode/georay"
# !pip install -q trimesh

# ==============================================================================
# Cell 2: Environment check
# ==============================================================================
import torch
import warp as wp

assert torch.cuda.is_available(), "Please enable GPU runtime!"
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import georay
print(f"GeoRay version: {georay.__version__}")
print(f"WarpTracer:     {'available' if georay.WarpTracer else 'NOT available'}")

# ==============================================================================
# Cell 3: Quick trace example
# ==============================================================================
import trimesh
from georay import WarpTracer, generate_rays

device = torch.device("cuda")

# Scene: glass cube
cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
fm[:, 0] = 0
fm[:, 1] = 1
materials = {
    0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
}

# Rays: incident from air along +z
rays = generate_rays(
    source_type="grid", num_rays=1024,
    origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.5, 0.5, -0.5, 0.5),
    polarization="s", device=device)

# Trace
tracer = WarpTracer(verts, faces, fm, materials, device)
out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

print(f"positions:        {tuple(out['positions'].shape)}")
print(f"energy_conserved: {out['energy_conserved']}")

# ==============================================================================
# Cell 4: Emit rays from inside a medium (new feature)
# ==============================================================================
from georay import generate_rays

# Start from inside glass at z=0
rays_inside = generate_rays(
    source_type="grid", num_rays=256,
    origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.3, 0.3, -0.3, 0.3),
    polarization="s",
    medium_id=1,          # Start inside glass (material 1)
    device=device)

out_inside = tracer.trace(rays_inside, max_steps=10)
print(f"Trace result from inside a medium:")
print(f"  energy_conserved: {out_inside['energy_conserved']}")
print(f"  n1[step 1] = {out_inside['n1'][0, 1, 0].item():.3f}  (expected 1.5)")

# ==============================================================================
# Cell 5: Exception handling example
# ==============================================================================
from georay import InvalidMaterialError, InvalidRayError

# Invalid material -> caught immediately
try:
    bad_materials = {0: {"n": -1.5}}   # n < 0
    WarpTracer(verts, faces, fm, bad_materials, device)
except InvalidMaterialError as e:
    print(f"Caught InvalidMaterialError: {e}")

# Invalid ray dict -> caught immediately
try:
    bad_rays = {
        "origins": torch.zeros((4, 3), device=device),
        # missing 'dirs' and other required fields
    }
    tracer.trace(bad_rays)
except InvalidRayError as e:
    print(f"Caught InvalidRayError: {e}")

# ==============================================================================
# Cell 6: Performance benchmark
# ==============================================================================
import time

for N in [256, 1024, 4096, 16384]:
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer.trace(rays, max_steps=2)   # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tracer.trace(rays, max_steps=20)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"  N={N:6d}: {t_ms:8.2f} ms  ({N/t_ms*1000/1e6:.2f} M rays/s)")
