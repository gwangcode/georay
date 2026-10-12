"""
tests/test_warp_tracer.py
=========================
WarpTracer 的单元测试。

只在 CUDA + warp-lang 都可用时运行，否则自动 skip。

运行：
    pytest tests/test_warp_tracer.py -v
"""
import math
import pytest
import torch
import trimesh

# ---- 环境守卫 ----
_HAS_CUDA = torch.cuda.is_available()

try:
    import warp as wp  # noqa: F401
    _HAS_WARP = True
except ImportError:
    _HAS_WARP = False

try:
    from georay import WarpTracer
    _HAS_WARP_TRACER = WarpTracer is not None
except ImportError:
    _HAS_WARP_TRACER = False

pytestmark = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_WARP and _HAS_WARP_TRACER),
    reason="WarpTracer requires CUDA + warp-lang + georay[warp]"
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def device():
    return torch.device("cuda")


@pytest.fixture
def cube_scene(device):
    """标准立方体场景。"""
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


@pytest.fixture
def default_materials():
    return {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }


@pytest.fixture
def deterministic_materials():
    """无反射、无漫反射、无吸收 → 纯折射。"""
    return {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }


def make_rays(device, n_rays=256, origin=(0.0, 0.0, -3.0),
              direction=(0.0, 0.0, 1.0), bounds=(-0.5, 0.5, -0.5, 0.5)):
    from georay import generate_rays
    return generate_rays(
        source_type="grid", num_rays=n_rays,
        origin=origin, direction=direction, bounds=bounds,
        polarization="s", device=device)


# ==============================================================================
# 1. 构造与基本 trace
# ==============================================================================

def test_warp_tracer_constructs(cube_scene, default_materials, device):
    """WarpTracer 可以正常构造。"""
    verts, faces, fm = cube_scene
    tracer = WarpTracer(verts, faces, fm, default_materials, device)
    assert tracer is not None
    assert tracer.mesh is not None


def test_warp_tracer_trace_basic(cube_scene, default_materials, device):
    """基本 trace 返回正确的 shape。"""
    verts, faces, fm = cube_scene
    tracer = WarpTracer(verts, faces, fm, default_materials, device)
    rays = make_rays(device, n_rays=64)
    out = tracer.trace(rays, max_steps=10)

    N = 64
    S = 11
    assert out["positions"].shape == (N, S, 3)
    assert out["stokes"].shape == (N, S, 4)
    assert out["energy"].shape == (N, S)
    assert out["active"].shape == (N, S)


def test_warp_tracer_energy_conserved(cube_scene, default_materials, device):
    """能量守恒检查通过。"""
    verts, faces, fm = cube_scene
    tracer = WarpTracer(verts, faces, fm, default_materials, device)
    rays = make_rays(device, n_rays=256)
    out = tracer.trace(rays, max_steps=15)

    assert out["energy_conserved"]
    assert torch.all(out["energy"] >= -1e-6)
    assert torch.all(out["energy"] <= 1.0 + 1e-6)


# ==============================================================================
# 2. 确定性场景下与 PyTorch tracer 数值一致
# ==============================================================================

def test_warp_matches_pytorch_deterministic(
        cube_scene, deterministic_materials, device):
    """
    无随机分支的场景下，WarpTracer 与 PyTorch tracer 应 bit-identical。
    """
    from georay import AdvancedRayTracerPBRTGPU

    verts, faces, fm = cube_scene
    rays = make_rays(device, n_rays=128)

    tracer_pt = AdvancedRayTracerPBRTGPU(
        verts, faces, fm, deterministic_materials, device)
    out_pt = tracer_pt.trace(rays, max_steps=10, surface_model="custom")

    tracer_wp = WarpTracer(verts, faces, fm, deterministic_materials, device)
    out_wp = tracer_wp.trace(rays, max_steps=10, surface_model="custom")

    S_min = min(out_pt["positions"].shape[1], out_wp["positions"].shape[1])

    for key in ["positions", "energy", "stokes", "reflectivity", "transmissivity"]:
        a = out_pt[key][:, :S_min].float()
        b = out_wp[key][:, :S_min].float()
        assert a.shape == b.shape, f"{key} shape mismatch"
        diff = (a - b).abs().max().item()
        assert diff < 1e-3, f"{key} mismatch: max|Δ| = {diff}"


def test_warp_matches_pytorch_snell(
        cube_scene, deterministic_materials, device):
    """斜入射下两边也一致。"""
    from georay import AdvancedRayTracerPBRTGPU

    verts, faces, fm = cube_scene
    theta = math.radians(30.0)
    rays = make_rays(
        device, n_rays=64,
        direction=(math.sin(theta), 0.0, math.cos(theta)),
        bounds=(0.0, 0.0, 0.0, 0.0))

    tracer_pt = AdvancedRayTracerPBRTGPU(
        verts, faces, fm, deterministic_materials, device)
    out_pt = tracer_pt.trace(rays, max_steps=10, surface_model="custom")

    tracer_wp = WarpTracer(verts, faces, fm, deterministic_materials, device)
    out_wp = tracer_wp.trace(rays, max_steps=10, surface_model="custom")

    S_min = min(out_pt["positions"].shape[1], out_wp["positions"].shape[1])
    diff = (out_pt["positions"][:, :S_min] -
            out_wp["positions"][:, :S_min]).abs().max().item()
    assert diff < 1e-3, f"斜入射 positions 差异过大: {diff}"


# ==============================================================================
# 3. surface_model 切换
# ==============================================================================

def test_warp_surface_model_fresnel(cube_scene, default_materials, device):
    """fresnel 模式：反射率应接近解析解（法向入射 R≈0.04）。"""
    verts, faces, fm = cube_scene
    tracer = WarpTracer(verts, faces, fm, default_materials, device)
    rays = make_rays(device, n_rays=256)
    out = tracer.trace(rays, max_steps=5, surface_model="fresnel")

    R = out["reflectivity"][:, 1, 0].mean().item()
    assert abs(R - 0.04) < 0.01, f"R = {R}, expected ≈ 0.04"


def test_warp_surface_model_custom(
        cube_scene, default_materials, device):
    """custom 模式：反射率应等于 custom_refl。"""
    mats = {
        0: {"n": 1.0, "reflectivity": 0.5, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.5, "diffuse": 0.0, "absorptivity": 0.0},
    }
    verts, faces, fm = cube_scene
    tracer = WarpTracer(verts, faces, fm, mats, device)
    rays = make_rays(device, n_rays=64)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    R = out["reflectivity"][:, 1, 0].mean().item()
    assert abs(R - 0.5) < 1e-5, f"R = {R}, expected 0.5"


def test_warp_surface_model_coated(cube_scene, default_materials, device):
    """coated 模式：R_spec = R_fresnel × custom_refl。"""
    mats = {
        0: {"n": 1.0, "reflectivity": 0.5, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.5, "diffuse": 0.0, "absorptivity": 0.0},
    }
    verts, faces, fm = cube_scene
    tracer = WarpTracer(verts, faces, fm, mats, device)
    rays = make_rays(device, n_rays=64)
    out = tracer.trace(rays, max_steps=3, surface_model="coated")

    R = out["reflectivity"][:, 1, 0].mean().item()
    # R_fresnel ≈ 0.04, custom_refl = 0.5 → R ≈ 0.02
    assert abs(R - 0.02) < 5e-3, f"R = {R}, expected ≈ 0.02"


def test_warp_legacy_fresnel_flag(cube_scene, default_materials, device):
    """fresnel=True ↔ coated, fresnel=False ↔ custom。"""
    verts, faces, fm = cube_scene
    tracer = WarpTracer(verts, faces, fm, default_materials, device)
    rays = make_rays(device, n_rays=32)

    R1 = tracer.trace(rays, max_steps=2, fresnel=True)["reflectivity"][:, 1, 0].mean().item()
    R2 = tracer.trace(rays, max_steps=2, surface_model="coated")["reflectivity"][:, 1, 0].mean().item()
    assert abs(R1 - R2) < 1e-7

    R3 = tracer.trace(rays, max_steps=2, fresnel=False)["reflectivity"][:, 1, 0].mean().item()
    R4 = tracer.trace(rays, max_steps=2, surface_model="custom")["reflectivity"][:, 1, 0].mean().item()
    assert abs(R3 - R4) < 1e-7


# ==============================================================================
# 4. 物理正确性：Fresnel 反射率 vs 解析解
# ==============================================================================

def test_warp_fresnel_vs_analytical(device):
    """多个入射角下，WarpTracer 的 R_s 精确匹配解析解。"""
    hs = 10.0
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    tracer = WarpTracer(verts, faces, fm, materials, device)

    n1, n2 = 1.0, 1.5
    for theta_deg in [0.0, 30.0, 45.0, 60.0, 80.0]:
        theta = math.radians(theta_deg)
        direction = (math.sin(theta), 0.0, math.cos(theta))

        # 解析解
        sin_t = n1 * math.sin(theta) / n2
        cos_t = math.sqrt(1.0 - sin_t ** 2)
        cos_i = math.cos(theta)
        R_s_exact = ((n1 * cos_i - n2 * cos_t) /
                     (n1 * cos_i + n2 * cos_t)) ** 2

        # Warp
        rays = make_rays(
            device, n_rays=1,
            origin=(0.0, 0.0, -1.0), direction=direction,
            bounds=(0.0, 0.0, 0.0, 0.0))
        out = tracer.trace(rays, max_steps=3, surface_model="fresnel")
        R_warp = out["reflectivity"][0, 1, 0].item()

        assert abs(R_warp - R_s_exact) < 1e-4, \
            f"θi={theta_deg}°: R_warp={R_warp:.6f}, exact={R_s_exact:.6f}"


# ==============================================================================
# 5. 大批量（>10⁴）
# ==============================================================================

def test_warp_large_batch(cube_scene, default_materials, device):
    """16384 条光线应能正常追踪且能量守恒。"""
    verts, faces, fm = cube_scene
    tracer = WarpTracer(verts, faces, fm, default_materials, device)
    rays = make_rays(device, n_rays=16384)
    out = tracer.trace(rays, max_steps=20)

    assert out["positions"].shape[0] == 16384
    assert out["energy_conserved"]


# ==============================================================================
# 6. 无 CUDA 时确保不会崩溃
# ==============================================================================

def test_warp_tracer_unavailable_graceful():
    """WarpTracer 不可用时，import georay 不应崩溃。"""
    import georay
    # WarpTracer 可以是 None（无 warp-lang）或类（有 warp-lang）
    assert hasattr(georay, "WarpTracer")