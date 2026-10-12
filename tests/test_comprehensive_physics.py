"""
tests/test_comprehensive_physics.py
====================================
Comprehensive physical validation of GeoRay beyond the basics.

Covers:
    1.  Multiple bounces between parallel partial mirrors
    2.  Diffuse hemisphere sampling (cosine-weighted distribution)
    3.  Dispersive media (wavelength-dependent n → different Snell angle)
    4.  GRIN gradient analytical verification
    5.  DictGridGRINField trilinear interpolation
    6.  Glass → air Fresnel energy conservation (below critical angle)
    7.  Grazing incidence numerical stability
    8.  Non-air/glass interface (glass → higher-index material)
    9.  Legacy `fresnel` flag semantics
   10.  Russian roulette smoke test
   11.  max_steps = 0 boundary
   12.  verbose mode smoke test

Run:
    pytest tests/test_comprehensive_physics.py -v
"""
import math
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays, parse_grin_input_to_field


# ==============================================================================
# Helpers
# ==============================================================================

def make_flat_interface(device, half_size=10.0):
    hs = half_size
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def make_box(extents, device):
    box = trimesh.creation.box(extents=extents)
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    return verts, faces


def unit(v):
    return v / (v.norm() + 1e-12)


def make_two_parallel_mirrors(device, half_size=5.0, z0=0.0, z1=1.0):
    """Two horizontal planes at z=z0 and z=z1, normals both +z."""
    hs = half_size
    verts = torch.tensor([
        [-hs, -hs, z0], [hs, -hs, z0], [hs, hs, z0], [-hs, hs, z0],
        [-hs, -hs, z1], [hs, -hs, z1], [hs, hs, z1], [-hs, hs, z1],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)
    return verts, faces


# ==============================================================================
# 1. Multiple bounces between parallel partial mirrors
# ==============================================================================

def test_multiple_bounces_between_parallel_mirrors():
    """
    Light bouncing between two parallel partial mirrors must:
      - keep energy in [0, 1]
      - keep energy monotonically non-increasing
      - multiply weight by ≤ custom_refl at each reflection
    """
    device = torch.device("cpu")
    verts, faces = make_two_parallel_mirrors(device)
    face_mats = torch.zeros((4, 2), dtype=torch.long, device=device)

    R = 0.9
    materials = {
        0: {"n": 1.0, "reflectivity": R, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=10, surface_model="custom", ds_init=0.05)

    energy = out["energy"][0]

    assert torch.all((energy >= -1e-6) & (energy <= 1.0 + 1e-6)), \
        f"energy out of [0,1]: {energy.tolist()}"

    diffs = energy[1:] - energy[:-1]
    assert torch.all(diffs <= 1e-6), f"energy increased: {energy.tolist()}"

    ratios = energy[1:] / (energy[:-1] + 1e-12)
    changed = ratios < 1.0 - 1e-6
    if changed.any():
        assert torch.all(ratios[changed] <= R + 1e-5), \
            f"jump ratio exceeds R={R}: {ratios.tolist()}"

    assert out["energy_conserved"]


# ==============================================================================
# 2. Diffuse hemisphere sampling
# ==============================================================================

def test_diffuse_hemisphere_sampling():
    """
    Pure diffuse reflection samples a cosine-weighted hemisphere.
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 1.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 1.0, "absorptivity": 0.0},
    }

    N = 1000
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.01, 0.01, -0.01, 0.01),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom", ds_init=0.5)

    p1 = out["positions"][:, 1]
    p2 = out["positions"][:, 2]
    active = out["active"][:, 1]

    d = p2 - p1
    norms = d.norm(dim=-1, keepdim=True)
    valid = active & (norms.squeeze(-1) > 1e-3)
    d_unit = d[valid] / (norms[valid] + 1e-12)

    assert torch.all(d_unit[:, 2] < 0), \
        f"diffuse direction not in -z hemisphere: {d_unit[d_unit[:, 2] >= 0][:3]}"

    cos_theta = -d_unit[:, 2]
    mean_cos = cos_theta.mean().item()
    assert abs(mean_cos - 2.0 / 3.0) < 0.05, \
        f"cosine-weighted distribution expected mean 2/3, got {mean_cos:.4f}"

    dop_after = out["dop"][:, 1]
    dop_active = dop_after[active]
    assert torch.all(dop_active < 1e-5), \
        f"DoP after diffuse should be ~0, got max {dop_active.max().item()}"


# ==============================================================================
# 3. Dispersive medium — wavelength-dependent Snell
# ==============================================================================

def test_dispersive_medium_snell():
    """
    A dispersive glass with n(400nm)=1.55 and n(700nm)=1.50 must give
    different refraction angles at 45° incidence.
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)

    glass_n = {"wavelengths_nm": [400.0, 700.0], "values": [1.55, 1.50]}
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": glass_n, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(45.0)
    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))

    def measure(wl):
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=direction,
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization="s", device=device, wavelength_nm=wl)
        tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
        out = tracer.trace(rays, max_steps=4, fresnel=False, ds_init=0.05)
        pos = out["positions"][0]
        d_before = unit(pos[1] - pos[0])
        d_after = unit(pos[2] - pos[1])
        cos_t = abs(d_after[2].item())
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t ** 2))
        return math.degrees(math.asin(sin_t))

    theta_400 = measure(400.0)
    theta_700 = measure(700.0)

    exp_400 = math.degrees(math.asin(math.sin(theta_i) / 1.55))
    exp_700 = math.degrees(math.asin(math.sin(theta_i) / 1.50))

    assert abs(theta_400 - exp_400) < 0.5, \
        f"400nm: got {theta_400:.3f}°, expected {exp_400:.3f}°"
    assert abs(theta_700 - exp_700) < 0.5, \
        f"700nm: got {theta_700:.3f}°, expected {exp_700:.3f}°"

    assert theta_700 > theta_400, \
        f"higher-wavelength light should refract less: {theta_400:.3f}° vs {theta_700:.3f}°"


# ==============================================================================
# 4. GRIN gradient analytical verification
# ==============================================================================

def test_grin_gradient_analytical():
    """
    n(x,y,z) = 1.5 + 0.2x + 0.1y + 0.05z → grad n = (0.2, 0.1, 0.05).
    """
    device = torch.device("cpu")

    def linear_grin(x, y, z, wl):
        return 1.5 + 0.2 * x + 0.1 * y + 0.05 * z

    field = parse_grin_input_to_field(linear_grin, device)
    r = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-0.5, 0.3, -0.2],
    ], device=device)

    n, grad = field(r)

    assert n.shape == (3, 1)
    assert grad.shape == (3, 3)

    # ★ 修正：手算第三个点：
    #   1.5 + 0.2*(-0.5) + 0.1*(0.3) + 0.05*(-0.2)
    # = 1.5 - 0.1 + 0.03 - 0.01 = 1.42
    expected_n = torch.tensor([[1.5], [1.85], [1.42]], device=device)
    assert torch.allclose(n, expected_n, atol=1e-5), \
        f"n mismatch: {n.tolist()} vs {expected_n.tolist()}"

    expected_grad = torch.tensor([
        [0.2, 0.1, 0.05],
        [0.2, 0.1, 0.05],
        [0.2, 0.1, 0.05],
    ], device=device)
    assert torch.allclose(grad, expected_grad, atol=1e-5), \
        f"gradient mismatch: {grad.tolist()}"


# ==============================================================================
# 5. DictGridGRINField trilinear interpolation
# ==============================================================================

def test_dict_grid_grin_interpolation():
    """
    A 1D-in-x dict GRIN field must interpolate linearly.
    """
    device = torch.device("cpu")

    cfg = {
        "x": [0.0, 1.0],
        "y": [0.0],
        "z": [0.0],
        "values": [[[1.5]], [[1.6]]],
    }
    field = parse_grin_input_to_field(cfg, device)

    r = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], device=device)
    n, grad = field(r)

    assert abs(n[0, 0].item() - 1.5) < 1e-5
    assert abs(n[1, 0].item() - 1.55) < 1e-5
    assert abs(n[2, 0].item() - 1.6) < 1e-5

    assert abs(grad[1, 0].item() - 0.1) < 1e-3, \
        f"grad_x should be ~0.1, got {grad[1, 0].item()}"
    assert abs(grad[1, 1].item()) < 1e-3
    assert abs(grad[1, 2].item()) < 1e-3


# ==============================================================================
# 6. Glass → air Fresnel energy conservation
# ==============================================================================

def test_glass_to_air_fresnel_energy():
    """
    Below critical angle, glass → air transmission must satisfy R + T + A = 1,
    and R must match the analytical Fresnel reflectance for s-polarization.
    """
    device = torch.device("cpu")
    n_glass, n_air = 1.5, 1.0

    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n_air, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_glass, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(30.0)   # < θc ≈ 41.8°
    direction = (math.sin(theta_i), 0.0, -math.cos(theta_i))

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 1.0), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=4, fresnel=True, ds_init=0.05)

    R = out["reflectivity"][0, 1, 0].item()
    T = out["transmissivity"][0, 1, 0].item()
    A = out["absorptivity"][0, 1, 0].item()

    assert abs(R + T + A - 1.0) < 1e-4, \
        f"R + T + A = {R + T + A:.8f} (should be 1)"

    # ★ 修正：用解析 Fresnel 公式而非硬编码阈值
    sin_t = n_glass * math.sin(theta_i) / n_air
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    R_s_expected = (
        (n_glass * cos_i - n_air * cos_t) /
        (n_glass * cos_i + n_air * cos_t)
    ) ** 2
    assert abs(R - R_s_expected) < 1e-4, \
        f"R = {R:.6f}, analytical R_s = {R_s_expected:.6f}"


# ==============================================================================
# 7. Grazing incidence numerical stability
# ==============================================================================

def test_grazing_incidence_stability():
    """
    At θi = 89.5°, R must approach 1 and the computation must not produce NaN.

    两个关键的数值稳定技巧：
      1) 让射线延长线穿过平面的内部点 (target_x, 0, 0)，而不是
         三角形顶点或边界 —— 避免 float32 下 u/v 边界判定被舍入误伤。
      2) 起点离平面只有 1mm，而不是 1m —— 这样 ray-AABB 的
         t_near / t_far 都很小，float32 的精度绰绰有余。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(89.5)
    target_x = 0.5          # 命中点 x 坐标（远离三角形顶点 / 边界）
    z_start = -0.001        # 起点距平面 1mm

    # 反推 x_start，使得射线延长线恰好过 (target_x, 0, 0)
    #   x_start + t*sin(θ) = target_x
    #   z_start + t*cos(θ) = 0
    #  ⇒ x_start = target_x - |z_start| * tan(θ)
    x_start = target_x - abs(z_start) * math.tan(theta_i)
    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(x_start, 0.0, z_start), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=4, fresnel=True, ds_init=0.005)

    R = out["reflectivity"][0, 1, 0].item()
    T = out["transmissivity"][0, 1, 0].item()

    # 数值健全性
    assert not math.isnan(R), "R is NaN at grazing incidence"
    assert not math.isnan(T), "T is NaN at grazing incidence"

    # 与解析 Fresnel 公式对比（s 偏振，n1=1.0, n2=1.5）
    sin_t = math.sin(theta_i) / 1.5
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    R_s_expected = (
        (1.0 * cos_i - 1.5 * cos_t) /
        (1.0 * cos_i + 1.5 * cos_t)
    ) ** 2

    assert R > 0.9, \
        f"R at 89.5° should approach 1, got {R:.6f}"
    assert abs(R - R_s_expected) < 5e-3, \
        f"R = {R:.6f}, analytical R_s = {R_s_expected:.6f}"
    assert abs(R + T - 1.0) < 1e-4, \
        f"R + T = {R + T:.8f}, should be 1"


# ==============================================================================
# 8. Non-air/glass interface (glass → higher-index)
# ==============================================================================

def test_non_air_glass_interface():
    """
    Normal incidence between two non-air media (n1=1.5 → n2=1.7):
    R = ((n1 - n2)/(n1 + n2))².
    """
    device = torch.device("cpu")
    n1, n2 = 1.5, 1.7
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
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=3, fresnel=True, ds_init=0.05)

    R = out["reflectivity"][0, 1, 0].item()
    assert abs(R - R_expected) < 1e-4, \
        f"R = {R:.8f}, expected {R_expected:.8f}"

    pos = out["positions"][0]
    d_after = unit(pos[2] - pos[1])
    assert abs(d_after[0].item()) < 1e-4
    assert abs(d_after[1].item()) < 1e-4


# ==============================================================================
# 9. Legacy fresnel flag semantics
# ==============================================================================

def test_legacy_fresnel_flag_mapping():
    """
    fresnel=True must equal surface_model='coated'
    fresnel=False must equal surface_model='custom'
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.5, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.5, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)

    R1 = tracer.trace(rays, max_steps=2, fresnel=True)["reflectivity"][0, 1, 0].item()
    R2 = tracer.trace(rays, max_steps=2, surface_model="coated")["reflectivity"][0, 1, 0].item()
    assert abs(R1 - R2) < 1e-7, f"fresnel=True vs 'coated': {R1} vs {R2}"

    R3 = tracer.trace(rays, max_steps=2, fresnel=False)["reflectivity"][0, 1, 0].item()
    R4 = tracer.trace(rays, max_steps=2, surface_model="custom")["reflectivity"][0, 1, 0].item()
    assert abs(R3 - R4) < 1e-7, f"fresnel=False vs 'custom': {R3} vs {R4}"

    assert abs(R1 - R3) > 1e-3


# ==============================================================================
# 10. Russian roulette smoke test
# ==============================================================================

def test_russian_roulette_smoke():
    """
    Enabling RR must not break invariants (energy in [0,1], monotonic).
    """
    device = torch.device("cpu")
    verts, faces = make_two_parallel_mirrors(device)
    face_mats = torch.zeros((4, 2), dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.5, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=200,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(-0.01, 0.01, -0.01, 0.01),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=10, surface_model="custom",
                       ds_init=0.05, rr_threshold=0.05)

    energy = out["energy"]
    assert torch.all((energy >= -1e-6) & (energy <= 1.0 + 1e-6))
    diffs = energy[:, 1:] - energy[:, :-1]
    assert torch.all(diffs <= 1e-6)
    assert out["energy_conserved"]


# ==============================================================================
# 11. max_steps = 0 boundary
# ==============================================================================

def test_max_steps_zero():
    """max_steps=0 returns exactly one history entry (the initial state)."""
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    face_mats = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # ★ 修正：grid 光源会对 num_rays 做平方数取整，用 4 而不是 3
    rays = generate_rays(
        source_type="grid", num_rays=4,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.01, 0.01, -0.01, 0.01),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    out = tracer.trace(rays, max_steps=0)

    assert out["positions"].shape == (4, 1, 3)
    assert out["energy"].shape == (4, 1)
    assert out["stokes"].shape == (4, 1, 4)
    assert torch.all(out["active"][:, 0])
    assert out["energy_conserved"]


# ==============================================================================
# 12. verbose mode smoke test
# ==============================================================================

def test_verbose_mode(capsys):
    """verbose=True must run without crashing and emit GRIN diagnostics."""
    device = torch.device("cpu")
    verts, faces = make_box((1.0, 1.0, 4.0), device)
    face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    face_mats[:, 0] = 0
    face_mats[:, 1] = 1

    def parabolic_grin(x, y, z, wl):
        return 1.5 - 0.1 * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": parabolic_grin},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.3, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials, device)
    _ = tracer.trace(rays, max_steps=50, ds_init=0.05,
                     ds_max=0.1, verbose=True)

    captured = capsys.readouterr()
    assert "GRIN" in captured.out, \
        f"verbose output did not contain 'GRIN': {captured.out[:200]}"


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))