"""
tests/test_physics.py
=====================
Physical validation of GeoRay against analytical solutions.

All tests compare the ray tracer's output to closed-form physics formulas.
An error in any core algorithm (Snell, Fresnel, TIR, GRIN, energy) will be
caught by at least one test.

Test list:
    1. test_straight_line_in_uniform_medium
    2. test_snell_law_refraction_angle
    3. test_fresnel_normal_incidence
    4. test_fresnel_angle_dependent_s_and_p
    5. test_total_internal_reflection
    6. test_brewster_angle
    7. test_energy_conservation
    8. test_grin_parabolic_lens_focuses

Run:
    pytest tests/test_physics.py -v
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
    Single flat interface at z=0. Triangle normal = +z.

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


def make_box(extents, device):
    """Axis-aligned box centered at origin."""
    box = trimesh.creation.box(extents=extents)
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    return verts, faces


def unit(v):
    return v / (v.norm() + 1e-12)


def fresnel_Rs_Rp(n1, n2, theta_i):
    """
    Analytical Fresnel reflectance for a plane interface.

    Returns (R_s, R_p) or (1.0, 1.0) if TIR.
    """
    sin_t = n1 * math.sin(theta_i) / n2
    if sin_t > 1.0:
        return 1.0, 1.0   # TIR
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    R_s = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    R_p = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2
    return R_s, R_p


# ==============================================================================
# Test 1: Straight-line propagation in a uniform medium
# ==============================================================================

def test_straight_line_in_uniform_medium():
    """
    A ray inside a uniform medium must not bend.

    Physical fact: d/ds(n dr/ds) = 0 when n = const.
    """
    device = torch.device("cpu")
    n_glass = 1.5

    verts, faces = make_box((4.0, 4.0, 4.0), device)
    face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    face_mats[:, 0] = 0    # + normal side -> air
    face_mats[:, 1] = 1    # - normal side -> glass

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_glass, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    x0, y0 = 0.5, 0.3
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(x0, y0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization="random", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    # ds_init large enough, max_steps enough to cross the box
    out = tracer.trace(rays, max_steps=200, fresnel=False, ds_init=0.1)

    pos = out["positions"][0]     # [S+1, 3]
    # Normal incidence means NO refraction, so x and y stay constant.
    assert torch.allclose(pos[:, 0],
                          torch.full_like(pos[:, 0], x0), atol=1e-4), \
        f"x drifted: {pos[:, 0].tolist()}"
    assert torch.allclose(pos[:, 1],
                          torch.full_like(pos[:, 1], y0), atol=1e-4), \
        f"y drifted: {pos[:, 1].tolist()}"


# ==============================================================================
# Test 2: Snell's law of refraction
# ==============================================================================

def test_snell_law_refraction_angle():
    """
    Verify n1 sin θi = n2 sin θt at a flat interface.

    Geometry:
        materials: 0 = air (n=1.0), 1 = glass (n=1.5)
        face_mats = [[1, 0]]: +z side = glass, -z side = air
        Ray starts in air at z = -1, travels toward +z.
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_i = math.radians(30.0)

    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=5, fresnel=False, ds_init=0.05)

    pos = out["positions"][0]
    d_before = unit(pos[1] - pos[0])
    d_after = unit(pos[2] - pos[1])

    cos_i = abs(d_before[2].item())
    cos_t = abs(d_after[2].item())
    sin_i = math.sqrt(max(0.0, 1.0 - cos_i ** 2))
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t ** 2))

    lhs = n1 * sin_i
    rhs = n2 * sin_t
    rel_err = abs(lhs - rhs) / max(lhs, 1e-12)
    assert rel_err < 1e-3, \
        f"Snell's law violated: n1 sin θi = {lhs:.6f}, n2 sin θt = {rhs:.6f}"


# ==============================================================================
# Test 3: Fresnel reflectance at normal incidence
# ==============================================================================

def test_fresnel_normal_incidence():
    """
    At normal incidence, R = ((n1 - n2) / (n1 + n2))².

    For n1 = 1.0, n2 = 1.5: R = (0.5/2.5)² = 0.04.

    Geometry:
        materials: 0 = air, 1 = glass
        face_mats = [[1, 0]]: +z side = glass, -z side = air
        Ray from air (z = -1) toward +z.
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    R_expected = ((n1 - n2) / (n1 + n2)) ** 2

    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=4, fresnel=True, ds_init=0.05)

    R_measured = out["reflectivity"][0, 1, 0].item()
    assert abs(R_measured - R_expected) < 1e-5, \
        f"R = {R_measured:.8f}, expected {R_expected:.8f}"


# ==============================================================================
# Test 4: Angle-dependent Fresnel reflectance (s and p)
# ==============================================================================

def test_fresnel_angle_dependent_s_and_p():
    """
    At 60° incidence (air -> glass n=1.5), compare R_s and R_p to the
    Fresnel equations.

    Geometry:
        materials: 0 = air (1.0), 1 = glass (1.5)
        face_mats = [[1, 0]]: +z side = glass, -z side = air
        Ray starts at z = -1 in air, travels toward +z.
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_i = math.radians(60.0)
    R_s_exp, R_p_exp = fresnel_Rs_Rp(n1, n2, theta_i)

    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))

    for pol, R_exp in [("s", R_s_exp), ("p", R_p_exp)]:
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=direction,
            bounds=(-0.001, 0.001, -0.001, 0.001),
            polarization=pol, device=device)

        tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
        out = tracer.trace(rays, max_steps=4, fresnel=True, ds_init=0.05)

        R_meas = out["reflectivity"][0, 1, 0].item()
        assert abs(R_meas - R_exp) < 1e-3, \
            f"R_{pol} = {R_meas:.6f}, expected {R_exp:.6f}"


# ==============================================================================
# Test 5: Total internal reflection (TIR)
# ==============================================================================

def test_total_internal_reflection():
    """
    For n1 = 1.5, n2 = 1.0, θc = 41.81°.
    - θi = 35° → partial reflection (R < 1)
    - θi = 50° → total reflection (R ≈ 1)

    Geometry:
        materials: 0 = glass (1.5), 1 = air (1.0)
        face_mats = [[0, 1]]: +z side = glass, -z side = air
        Ray starts in glass at z = +1, travels toward -z.
    """
    device = torch.device("cpu")
    n1, n2 = 1.5, 1.0
    theta_c_deg = math.degrees(math.asin(n2 / n1))
    assert abs(theta_c_deg - 41.81) < 0.02

    verts, faces = make_flat_interface(device)
    # +z side = glass (mat 0), -z side = air (mat 1)
    face_mats = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    for theta_deg, expect_tir in [(35.0, False), (50.0, True)]:
        theta = math.radians(theta_deg)
        direction = (math.sin(theta), 0.0, -math.cos(theta))
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, 1.0), direction=direction,
            bounds=(-0.001, 0.001, -0.001, 0.001),
            polarization="s", device=device)

        tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
        out = tracer.trace(rays, max_steps=4, fresnel=True, ds_init=0.05)

        R = out["reflectivity"][0, 1, 0].item()
        if expect_tir:
            assert R > 0.99, f"TIR expected at {theta_deg}°, got R = {R:.6f}"
        else:
            assert R < 0.9, f"No TIR at {theta_deg}°, got R = {R:.6f}"


# ==============================================================================
# Test 6: Brewster angle (R_p = 0)
# ==============================================================================

def test_brewster_angle():
    """
    At the Brewster angle θ_B = atan(n2/n1), R_p = 0.

    For air → glass n=1.5: θ_B ≈ 56.31°.

    Geometry:
        materials: 0 = air, 1 = glass
        face_mats = [[1, 0]]: +z side = glass, -z side = air
        Ray starts at z = -1 in air, travels toward +z.
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_B_deg = math.degrees(math.atan(n2 / n1))

    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_B = math.radians(theta_B_deg)
    direction = (math.sin(theta_B), 0.0, math.cos(theta_B))

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization="p", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=4, fresnel=True, ds_init=0.05)

    R_p = out["reflectivity"][0, 1, 0].item()
    # Ideal R_p = 0; numerical floor 1e-2 accounts for finite angular resolution
    assert R_p < 1e-2, \
        f"At Brewster angle, R_p should be ~0, got {R_p:.8f}"


# ==============================================================================
# Test 7: Energy conservation (R + T + A = 1)
# ==============================================================================

def test_energy_conservation():
    """
    For non-absorbing media: R + T + A must equal 1 at every surface hit.
    """
    device = torch.device("cpu")

    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    for theta_deg in [0.0, 30.0, 60.0, 80.0]:
        theta = math.radians(theta_deg)
        direction = (math.sin(theta), 0.0, math.cos(theta))
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=direction,
            bounds=(-0.001, 0.001, -0.001, 0.001),
            polarization="s", device=device)

        tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
        out = tracer.trace(rays, max_steps=4, fresnel=True, ds_init=0.05)

        R = out["reflectivity"][0, 1, 0].item()
        T = out["transmissivity"][0, 1, 0].item()
        A = out["absorptivity"][0, 1, 0].item()
        total = R + T + A

        assert abs(total - 1.0) < 1e-4, \
            f"θi = {theta_deg}°: R + T + A = {total:.8f} (should be 1)"


# ==============================================================================
# Test 8: GRIN parabolic lens focuses rays
# ==============================================================================

def test_grin_parabolic_lens_focuses():
    """
    A parabolic GRIN medium n(r) = n0 - A·r² bends parallel rays toward
    the optical axis.

    A ray entering off-axis must have |x_final| < |x_initial|.
    """
    device = torch.device("cpu")
    n0 = 1.5
    A = 0.1    # weak enough that n stays positive everywhere in the box

    verts, faces = make_box((1.0, 1.0, 4.0), device)
    face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    face_mats[:, 0] = 0
    face_mats[:, 1] = 1

    def parabolic_grin(x, y, z, wavelength):
        return n0 - A * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n0, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": parabolic_grin},
    }

    x0 = 0.3
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(x0, 0.0, -3.0),      # start OUTSIDE the box
        direction=(0.0, 0.0, 1.0),
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization="random", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=400, ds_init=0.01,
                       ds_min=1e-4, ds_max=0.05, rk_tol=1e-5,
                       fresnel=False)

    pos = out["positions"][0]
    inside = pos[:, 2].abs() < 1.99
    inside_idx = torch.where(inside)[0]
    assert inside_idx.numel() > 0, "No points recorded inside the box"

    x_final = pos[inside_idx[-1], 0].item()
    # Ray must bend toward the axis
    assert abs(x_final) < abs(x0) - 0.01, \
        f"GRIN did not bend ray enough: x0 = {x0}, x_final = {x_final}"


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))