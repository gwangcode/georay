"""
tests/test_warp_tracer_extended.py
===================================
WarpTracer 的偏振、漫反射、吸收的扩展验证。
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, WarpTracer, generate_rays


_HAS_CUDA = torch.cuda.is_available()
try:
    import warp as wp
    _HAS_WARP = True
except ImportError:
    _HAS_WARP = False
try:
    from georay import WarpTracer as _WT
    _HAS_WARP_TRACER = _WT is not None
except ImportError:
    _HAS_WARP_TRACER = False


_skip_warp = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_WARP and _HAS_WARP_TRACER),
    reason="WarpTracer requires CUDA + warp-lang + georay[warp]"
)


# ==============================================================================
# 场景
# ==============================================================================

def make_flat_interface(device, hs=100.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


# ==============================================================================
# Test 1: Warp 偏振 vs PyTorch 偏振（确定性）
# ==============================================================================

@_skip_warp
def test_warp_polarization_matches_pytorch():
    """
    45° 线偏振，45° 入射角，纯透射界面。
    Warp 与 PyTorch 的 S1/S2 输出应完全一致。
    """
    device = torch.device("cuda")
    n1, n2 = 1.0, 1.5
    theta_i = math.radians(45.0)

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out_pt = tp.trace(rays, max_steps=3, surface_model="custom")

    tw = WarpTracer(verts, faces, fm, materials, device)
    out_wp = tw.trace(rays, max_steps=3, surface_model="custom")

    S1_pt = out_pt["stokes"][0, 1, 1].item()
    S2_pt = out_pt["stokes"][0, 1, 2].item()
    S1_wp = out_wp["stokes"][0, 1, 1].item()
    S2_wp = out_wp["stokes"][0, 1, 2].item()

    print("\n[Test 1] Warp vs PyTorch polarization")
    print("  PyTorch: S1={:.6f}, S2={:.6f}".format(S1_pt, S2_pt))
    print("  Warp   : S1={:.6f}, S2={:.6f}".format(S1_wp, S2_wp))

    assert abs(S1_pt - S1_wp) < 1e-5, "S1 mismatch"
    assert abs(S2_pt - S2_wp) < 1e-5, "S2 mismatch"


# ==============================================================================
# Test 2: Warp 漫反射半球采样
# ==============================================================================

@_skip_warp
def test_warp_diffuse_hemisphere():
    """
    纯漫反射表面，Warp 的余弦加权半球采样。
    统计检查：cos θ 均值 ≈ 2/3。
    """
    device = torch.device("cuda")
    hs = 10.0
    verts, faces = make_flat_interface(device, hs)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
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

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=3, surface_model="custom")

    # step 2 - step 1 的方向
    p1 = out["positions"][:, 1, :]
    p2 = out["positions"][:, 2, :]
    active = out["active"][:, 1]

    d = p2 - p1
    norms = d.norm(dim=-1, keepdim=True)
    valid = active & (norms.squeeze(-1) > 1e-3)
    d_unit = d[valid] / (norms[valid] + 1e-12)

    # 半球方向应指向 -z（反射回来）
    assert torch.all(d_unit[:, 2] < 0), \
        "Warp diffuse direction not in -z hemisphere"

    cos_theta = -d_unit[:, 2]
    mean_cos = cos_theta.mean().item()

    print("\n[Test 2] Warp diffuse hemisphere")
    print("  N = {}, mean cos θ = {:.4f} (expected ≈ 2/3)".format(N, mean_cos))

    assert abs(mean_cos - 2.0 / 3.0) < 0.05, \
        "Warp diffuse: mean cos θ = {:.4f}, expected ≈ 2/3".format(mean_cos)


# ==============================================================================
# Test 3: Warp 吸收
# ==============================================================================

@_skip_warp
def test_warp_absorption():
    """
    有吸收的界面：R + T + A = 1，且 A ≈ absorptivity。
    """
    device = torch.device("cuda")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.3},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.3},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=3, surface_model="fresnel")

    R = out["reflectivity"][0, 1, 0].item()
    T = out["transmissivity"][0, 1, 0].item()
    A = out["absorptivity"][0, 1, 0].item()

    print("\n[Test 3] Warp absorption")
    print("  R = {:.6f}, T = {:.6f}, A = {:.6f}, sum = {:.6f}".format(
        R, T, A, R + T + A))

    assert abs(R + T + A - 1.0) < 1e-4, "R+T+A != 1: {}".format(R + T + A)
    assert A > 0.2, "Absorptivity should be ~0.3, got {:.4f}".format(A)


# ==============================================================================
# Test 4: Warp 多次反弹 + 偏振累积
# ==============================================================================

@_skip_warp
def test_warp_multiple_bounce_polarization():
    """
    45° 线偏振在两个平行镜面间多次反射。
    Warp 的 S2 应在每次反射时翻转符号。
    """
    device = torch.device("cuda")
    hs = 5.0
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
        [-hs, -hs, 1.0], [hs, -hs, 1.0], [hs, hs, 1.0], [-hs, hs, 1.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)
    fm = torch.zeros((4, 2), dtype=torch.long, device=device)
    fm[0:2, 0] = 0
    fm[0:2, 1] = 1
    fm[2:4, 0] = 1
    fm[2:4, 1] = 0

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=4, surface_model="custom")

    S2_0 = out["stokes"][0, 0, 2].item()
    S2_1 = out["stokes"][0, 1, 2].item()
    S2_2 = out["stokes"][0, 2, 2].item()

    print("\n[Test 4] Warp multiple bounce polarization")
    print("  S2 initial: {:+.6f}".format(S2_0))
    print("  S2 after 1: {:+.6f}".format(S2_1))
    print("  S2 after 2: {:+.6f}".format(S2_2))

    assert abs(S2_0 - 1.0) < 1e-5, "Initial S2 must be +1"
    assert abs(S2_1 + 1.0) < 1e-5, "After 1 bounce S2 must be -1, got {}".format(S2_1)
    assert abs(S2_2 - 1.0) < 1e-5, "After 2 bounces S2 must be +1, got {}".format(S2_2)


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))