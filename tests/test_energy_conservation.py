import math
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def test_energy_weight_decreases_monotonically():
    """The Monte Carlo weight must be non-increasing along a ray."""
    device = torch.device("cpu")

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

    rays = generate_rays(
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

    # energy in [0, 1]
    energy = out["energy"]
    assert torch.all((energy >= -1e-6) & (energy <= 1.0 + 1e-6))

    # monotonic non-increasing
    diffs = energy[:, 1:] - energy[:, :-1]
    assert torch.all(diffs <= 1e-6)

    # energy_conserved flag
    assert out["energy_conserved"]


def test_surface_model_switching():
    """Different surface_models give different reflectance."""
    device = torch.device("cpu")

    verts = torch.tensor([
        [-10.0, -10.0, 0.0], [10.0, -10.0, 0.0],
        [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)

    # custom_refl = 0.5 (half), fresnel R_s ~ 0.04 (normal incidence)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.5, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.5, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)

    R_fresnel = tracer.trace(rays, max_steps=4,
                              surface_model="fresnel")["reflectivity"][0, 1, 0].item()
    R_custom = tracer.trace(rays, max_steps=4,
                             surface_model="custom")["reflectivity"][0, 1, 0].item()
    R_coated = tracer.trace(rays, max_steps=4,
                             surface_model="coated")["reflectivity"][0, 1, 0].item()

    # R_fresnel should be ~0.04
    assert abs(R_fresnel - 0.04) < 1e-3, f"R_fresnel = {R_fresnel}"
    # R_custom should be ~0.5 (pure custom)
    assert abs(R_custom - 0.5) < 1e-3, f"R_custom = {R_custom}"
    # R_coated should be ~0.04 * 0.5 = 0.02
    assert abs(R_coated - 0.02) < 1e-3, f"R_coated = {R_coated}"