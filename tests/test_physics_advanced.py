"""
tests/test_physics_advanced.py
==============================
Advanced physical validation of GeoRay.

Covers:
    1. Absorption in a lossy dielectric
    2. TIR phase retardation (polarization)
    3. Stokes vector normalization
    4. Cross-validation against a brute-force ray-plane computation

All tests use closed-form analytical references or explicit geometric
facts to verify that GeoRay's core physics is correct.

Run:
    pytest tests/test_physics_advanced.py -v
"""
import math
import numpy as np
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


# ==============================================================================
# Helpers
# ==============================================================================

def make_flat_interface(device, half_size=10.0):
    """
    Single flat interface at z=0.  Face normal = +z.

    face_mats[i] = [mat_+z, mat_-z]:
        face_mats[i, 0] = material on the +z side of the plane
        face_mats[i, 1] = material on the -z side of the plane
    """
    hs = half_size
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


# ==============================================================================
# Test 1: Absorption reduces energy (Beer-Lambert-like)
# ==============================================================================

def test_absorption_reduces_energy():
    """
    A lossy dielectric must absorb a fraction of the incident light.

    Geometry:
        materials: 0 = air (n=1.0), 1 = lossy glass (n=1.5, abs=0.3)
        face_mats = [[1, 0]]: +z side = lossy glass, -z side = air
        Ray starts at z = -1 in air, travels toward +z.
    """
    device = torch.device("cpu")

    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.3},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=4, fresnel=False, ds_init=0.05)

    # The absorptivity of material 1 should be reflected in the output.
    A = out["absorptivity"][0, 1, 0].item()
    assert A > 0.2, f"Absorptivity should be ~0.3, got {A:.4f}"

    # Energy conservation: R + T + A = 1
    R = out["reflectivity"][0, 1, 0].item()
    T = out["transmissivity"][0, 1, 0].item()
    assert abs(R + T + A - 1.0) < 1e-5, \
        f"R + T + A = {R + T + A:.8f} (should be 1)"


# ==============================================================================
# Test 2: TIR phase retardation (polarization)
# ==============================================================================

def test_tir_phase_retardation():
    """
    A 45° linear polarization entering TIR becomes elliptical.

    Under TIR, δ_s ≠ δ_p, so the relative phase between s and p
    components evolves.  For a 45° linear input (S1 = 0, S2 = 1, S3 = 0),
    TIR should produce a nonzero S3 IF the phase is tracked.

    GeoRay v1.1 uses REAL-valued Fresnel coefficients, so the phase is
    LOST and S3 remains ~0.  This test documents the current limitation.
    It is marked as a "documentation" test that passes regardless of S3,
    but prints which regime we are in.

    Geometry:
        materials: 0 = glass (n=1.5), 1 = air (n=1.0)
        face_mats = [[0, 1]]: +z side = glass, -z side = air
        Ray starts in glass at z = +1, travels toward -z at 50° (> θc).
    """
    device = torch.device("cpu")

    verts, faces = make_flat_interface(device)
    # +z side = glass (mat 0), -z side = air (mat 1)
    face_mats = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta = math.radians(50.0)   # > θc for n1=1.5, n2=1.0
    direction = (math.sin(theta), 0.0, -math.cos(theta))

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 1.0), direction=direction,
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=4, fresnel=True, ds_init=0.05)

    stokes_in = out["stokes"][0, 0]
    stokes_out = out["stokes"][0, 1]

    # `polarization=45.0` → Stokes = (1, cos 90°, sin 90°, 0) = (1, 0, 1, 0)
    # i.e., S1 = 0, S2 = 1, S3 = 0  (a 45° linear polarization)
    assert abs(stokes_in[1].item()) < 0.01, \
        f"S1 should be ~0, got {stokes_in[1].item()}"
    assert abs(stokes_in[2].item() - 1.0) < 0.01, \
        f"S2 should be ~1, got {stokes_in[2].item()}"
    assert abs(stokes_in[3].item()) < 0.01, \
        f"S3 should be ~0, got {stokes_in[3].item()}"

    # DoP must remain 1 (no depolarization at a smooth interface)
    dop = out["dop"][0, 1, 0].item()
    assert abs(dop - 1.0) < 1e-3, f"DoP changed: {dop}"

    # Document S3 behavior (real vs complex Fresnel)
    s3_out = stokes_out[3].item()
    if abs(s3_out) > 1e-4:
        print(f"\n  [TIR test] S3_out = {s3_out:+.6f}  (phase preserved)")
    else:
        print(f"\n  [TIR test] S3_out = {s3_out:+.6f}  "
              f"(phase lost — expected for GeoRay v1.1 real Fresnel)")


# ==============================================================================
# Test 3: Stokes vector normalization
# ==============================================================================

def test_stokes_normalization():
    """
    After each surface interaction:
        - S0 must remain equal to 1 (normalized)
        - DoP must stay in [0, 1]
    """
    device = torch.device("cpu")

    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    face_mats[:, 0] = 0    # outside = air
    face_mats[:, 1] = 1    # inside = glass

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
    out = tracer.trace(rays, max_steps=20, fresnel=True, ds_init=0.05)

    # S0 must be 1 for all active rays
    S0 = out["stokes"][..., 0]     # [N, S+1]
    active = out["active"]          # [N, S+1]
    S0_active = S0[active]
    assert torch.all((S0_active - 1.0).abs() < 1e-5), \
        f"S0 out of range: min={S0_active.min()}, max={S0_active.max()}"

    # DoP must be in [0, 1]
    dop = out["dop"]
    assert torch.all((dop >= -1e-6) & (dop <= 1.0 + 1e-6)), \
        f"DoP out of range: min={dop.min()}, max={dop.max()}"


# ==============================================================================
# Test 4: Cross-validation — analytical ray-plane intersection
# ==============================================================================

def test_cross_validation_brute_force_plane():
    """
    Cross-validate GeoRay's intersection distance against the analytical
    ray-plane formula.

    A flat interface at z = 0.  Ray starts at (0, 0, -1) going +z.
    The first hit must be at distance exactly 1.0.

    Geometry:
        materials: 0 = air, 1 = glass
        face_mats = [[1, 0]]: +z side = glass, -z side = air
        Ray from z = -1 along +z.
    """
    device = torch.device("cpu")

    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=4, fresnel=False, ds_init=0.05)

    # The recorded position at step 1 = first surface hit
    p1 = out["positions"][0, 1]
    dist_measured = (p1 - torch.tensor([0.0, 0.0, -1.0])).norm().item()

    assert abs(dist_measured - 1.0) < 1e-4, \
        f"Intersection distance = {dist_measured}, expected 1.0"

    # Hit point must be at z ≈ 0
    assert abs(p1[2].item()) < 1e-5, \
        f"Hit point z = {p1[2].item()}, expected 0"


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))