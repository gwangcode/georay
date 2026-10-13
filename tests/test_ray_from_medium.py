"""
tests/test_ray_from_medium.py
==============================
光线从介质内部出发的测试。
"""
import math
import inspect
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays

try:
    from georay import WarpTracer
    _HAS_WARP_TRACER = WarpTracer is not None
except ImportError:
    WarpTracer = None
    _HAS_WARP_TRACER = False

_HAS_CUDA = torch.cuda.is_available()


def _supports_medium_ids(cls):
    try:
        src = inspect.getsource(cls.trace)
    except (OSError, TypeError):
        return False
    return "medium_ids" in src


_HAS_MEDIUM_API = _supports_medium_ids(AdvancedRayTracerPBRTGPU)

_skip_if_no_api = pytest.mark.skipif(
    not _HAS_MEDIUM_API,
    reason="tracer.trace() does not yet support ray_dict['medium_ids']"
)


def make_thin_glass_slab(device):
    """薄玻璃板 z ∈ [-0.5, 0.5]，xy 方向很大 (10×10)。"""
    verts = torch.tensor([
        [-5.0, -5.0, -0.5], [5.0, -5.0, -0.5], [5.0, 5.0, -0.5], [-5.0, 5.0, -0.5],
        [-5.0, -5.0,  0.5], [5.0, -5.0,  0.5], [5.0, 5.0,  0.5], [-5.0, 5.0,  0.5],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)
    fm = torch.zeros((4, 2), dtype=torch.long, device=device)
    fm[0:2, 0] = 1   # z=-0.5 面：+z 侧是玻璃
    fm[0:2, 1] = 0   # -z 侧是空气
    fm[2:4, 0] = 0   # z=+0.5 面：+z 侧是空气
    fm[2:4, 1] = 1   # -z 侧是玻璃
    return verts, faces, fm


def make_glass_box(device):
    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


# ==============================================================================
# Test 1: 玻璃内部出发 → 从表面折射
# ==============================================================================

@_skip_if_no_api
def test_ray_from_glass_medium():
    device = torch.device("cpu")
    verts, faces, fm = make_thin_glass_slab(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays["medium_ids"] = torch.tensor([1], dtype=torch.long, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="custom")

    n1 = out["n1"][0, 1, 0].item()
    n2 = out["n2"][0, 1, 0].item()

    print("\n[Test 1] Ray from glass medium")
    print("  n1={:.4f}, n2={:.4f}".format(n1, n2))

    assert abs(n1 - 1.5) < 1e-4, "n1 should be 1.5, got {}".format(n1)
    assert abs(n2 - 1.0) < 1e-4, "n2 should be 1.0, got {}".format(n2)


# ==============================================================================
# Test 2: 从内部出发 → TIR（薄板）
# ==============================================================================

@_skip_if_no_api
def test_ray_from_medium_tir():
    """
    光线从薄玻璃板内部，以 50° (> θc ≈ 41.8°) 入射到 z=+0.5 界面。
    应发生 TIR。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_thin_glass_slab(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta = math.radians(50.0)
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.0),
        direction=(math.sin(theta), 0.0, math.cos(theta)),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays["medium_ids"] = torch.tensor([1], dtype=torch.long, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="fresnel")

    R = out["reflectivity"][0, 1, 0].item()

    print("\n[Test 2] Ray from glass medium → TIR")
    print("  theta = 50°, thin slab, R = {:.6f}".format(R))

    assert R > 0.99, "TIR expected, got R = {:.6f}".format(R)


# ==============================================================================
# Test 3: 从 GRIN 介质内部出发
# ==============================================================================

@_skip_if_no_api
def test_ray_from_grin_medium():
    device = torch.device("cpu")
    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1

    def parabolic_grin(x, y, z, wl):
        return 1.5 - 0.1 * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": parabolic_grin},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays["medium_ids"] = torch.tensor([1], dtype=torch.long, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=50, ds_init=0.02, fresnel=False)

    hist_refl = out["reflectivity"][0, :, 0]
    hist_trans = out["transmissivity"][0, :, 0]

    print("\n[Test 3] Ray from GRIN medium")
    print("  step 1: refl={:.4f}, trans={:.4f}".format(
        hist_refl[1].item(), hist_trans[1].item()))

    assert hist_refl[1].item() < 1e-6, "GRIN step should have no reflection"
    assert hist_trans[1].item() < 1e-6, "GRIN step should have no transmission"


# ==============================================================================
# Test 4: 两条光线从不同介质同时出发
# ==============================================================================

@_skip_if_no_api
def test_two_rays_from_different_media():
    device = torch.device("cpu")
    verts, faces, fm = make_thin_glass_slab(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays_air = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays_air["medium_ids"] = torch.tensor([0], dtype=torch.long, device=device)

    rays_glass = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays_glass["medium_ids"] = torch.tensor([1], dtype=torch.long, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    out_air = tracer.trace(rays_air, max_steps=5, surface_model="custom")
    out_glass = tracer.trace(rays_glass, max_steps=5, surface_model="custom")

    n1_air = out_air["n1"][0, 1, 0].item()
    n2_air = out_air["n2"][0, 1, 0].item()
    n1_glass = out_glass["n1"][0, 1, 0].item()
    n2_glass = out_glass["n2"][0, 1, 0].item()

    print("\n[Test 4] Two rays from different media")
    print("  From air  : n1={:.4f} → n2={:.4f}".format(n1_air, n2_air))
    print("  From glass: n1={:.4f} → n2={:.4f}".format(n1_glass, n2_glass))

    assert abs(n1_air - 1.0) < 1e-4 and abs(n2_air - 1.5) < 1e-4
    assert abs(n1_glass - 1.5) < 1e-4 and abs(n2_glass - 1.0) < 1e-4


# ==============================================================================
# Test 5: WarpTracer 从介质内部出发
# ==============================================================================

@pytest.mark.skipif(not (_HAS_CUDA and _HAS_WARP_TRACER),
                     reason="WarpTracer requires CUDA + warp-lang")
def test_warp_ray_inside_glass():
    device = torch.device("cuda")
    verts, faces, fm = make_thin_glass_slab(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays["medium_ids"] = torch.tensor([1], dtype=torch.long, device=device)

    tracer = WarpTracer(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="custom")

    n1 = out["n1"][0, 1, 0].item()
    n2 = out["n2"][0, 1, 0].item()

    print("\n[Test 5] WarpTracer from inside glass")
    print("  n1 = {:.4f}, n2 = {:.4f}".format(n1, n2))

    assert abs(n1 - 1.5) < 1e-3
    assert abs(n2 - 1.0) < 1e-3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))