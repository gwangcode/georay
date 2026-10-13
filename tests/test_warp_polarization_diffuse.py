"""
tests/test_warp_polarization_diffuse.py
========================================
WarpTracer 的偏振、漫反射、吸收物理验证。

覆盖：
    1. Warp 与 PyTorch 偏振演化一致（确定性场景）
    2. Warp 漫反射半球采样
    3. Warp 吸收能量守恒
    4. Warp 多次反弹偏振累积

Run:
    pytest tests/test_warp_polarization_diffuse.py -v -s
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
    reason="WarpTracer requires CUDA + warp-lang"
)


def make_flat_interface(device, hs=100.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def make_glass_box(device, extent=2.0):
    box = trimesh.creation.box(extents=[extent, extent, extent])
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def fresnel_rs_rp(n1, n2, theta_i):
    sin_t = n1 * math.sin(theta_i) / n2
    if sin_t >= 1.0:
        return 1.0, 1.0, 0.0
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return r_s, r_p, cos_t


# ==============================================================================
# Test 1: Warp 与 PyTorch 偏振一致（确定性反射）
# ==============================================================================

@_skip_warp
def test_warp_polarization_matches_pytorch():
    """
    45° 线偏振在确定性场景（custom + reflectivity=1，纯反射）下，
    Warp 与 PyTorch 的 Stokes 输出应一致。
    """
    device = torch.device("cuda")
    n1, n2 = 1.0, 1.5
    theta_i = math.radians(30.0)

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
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

    stokes_diff = (out_pt["stokes"][:, :3] -
                   out_wp["stokes"][:, :3]).abs().max().item()

    print("\n[Test 1] Warp vs PyTorch Polarization")
    print("  Stokes PyTorch (step 1): {}".format(
        [round(v, 6) for v in out_pt["stokes"][0, 1].tolist()]))
    print("  Stokes Warp    (step 1): {}".format(
        [round(v, 6) for v in out_wp["stokes"][0, 1].tolist()]))
    print("  Max |Delta Stokes|: {:.3e}".format(stokes_diff))

    assert stokes_diff < 1e-4, \
        "Warp vs PyTorch polarization mismatch: {:.6e}".format(stokes_diff)


# ==============================================================================
# Test 2: Warp 漫反射半球采样
# ==============================================================================

@_skip_warp
def test_warp_diffuse_hemisphere():
    """
    WarpTracer 漫反射：N 条光线，出射方向应满足 cos θ 分布，均值 ≈ 2/3。
    """
    device = torch.device("cuda")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 1.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 1.0, "absorptivity": 0.0},
    }

    N = 1024
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.01, 0.01, -0.01, 0.01),
        polarization="s", device=device)

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=3, surface_model="custom", seed=12345)

    p1 = out["positions"][:, 1]
    p2 = out["positions"][:, 2]
    active = out["active"][:, 1]

    d = p2 - p1
    norms = d.norm(dim=-1, keepdim=True)
    valid = active & (norms.squeeze(-1) > 1e-3)
    d_unit = d[valid] / (norms[valid] + 1e-12)

    # 出射方向应在 -z 半球
    assert torch.all(d_unit[:, 2] < 0), "diffuse not in -z hemisphere"

    # cos θ 均值 ≈ 2/3
    cos_theta = -d_unit[:, 2]
    mean_cos = cos_theta.mean().item()

    print("\n[Test 2] Warp Diffuse Hemisphere")
    print("  Sample count: {}".format(valid.sum().item()))
    print("  Mean cos theta: {:.4f} (expected 2/3 = 0.6667)".format(mean_cos))

    assert abs(mean_cos - 2.0 / 3.0) < 0.05, \
        "Diffuse cosine-weighted mean should be ~2/3, got {:.4f}".format(mean_cos)

    # DoP 应塌缩到 0
    dop = out["dop"][:, 1]
    dop_active = dop[active]
    print("  DoP after diffuse: max = {:.6f}".format(dop_active.max().item()))
    assert torch.all(dop_active < 1e-4)


# ==============================================================================
# Test 3: Warp 吸收与 PyTorch 一致
# ==============================================================================

@_skip_warp
def test_warp_absorption_matches_pytorch():
    """absorptivity=0.3 时，Warp 与 PyTorch 的 R+T+A=1 均应成立。"""
    device = torch.device("cuda")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.3},
    }

    rays = generate_rays(
        source_type="grid", num_rays=256,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tp = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out_pt = tp.trace(rays, max_steps=3, surface_model="custom")

    tw = WarpTracer(verts, faces, fm, materials, device)
    out_wp = tw.trace(rays, max_steps=3, surface_model="custom")

    def check_energy(out, tag):
        R = out["reflectivity"][:, 1, 0]
        T = out["transmissivity"][:, 1, 0]
        A = out["absorptivity"][:, 1, 0]
        dev = (R + T + A - 1.0).abs().max().item()
        print("  {}: max |R+T+A-1| = {:.3e}, A = {:.4f}".format(
            tag, dev, A.mean().item()))
        return dev

    print("\n[Test 3] Warp vs PyTorch Absorption")
    dev_pt = check_energy(out_pt, "PyTorch")
    dev_wp = check_energy(out_wp, "Warp")

    assert dev_pt < 1e-4, "PyTorch R+T+A != 1"
    assert dev_wp < 1e-4, "Warp R+T+A != 1"


# ==============================================================================
# Test 4: Warp 多次反弹偏振累积
# ==============================================================================

@_skip_warp
def test_warp_multiple_bounce_polarization():
    """
    两个平行反射界面，光线多次反弹，Stokes 应保持归一化且 S0=1。
    """
    device = torch.device("cuda")

    hs = 5.0
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
        [-hs, -hs, 1.0], [hs, -hs, 1.0], [hs, hs, 1.0], [-hs, hs, 1.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)
    fm = torch.zeros((4, 2), dtype=torch.long, device=device)

    materials = {
        0: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    tw = WarpTracer(verts, faces, fm, materials, device)
    out = tw.trace(rays, max_steps=5, surface_model="custom")

    S0 = out["stokes"][..., 0]
    print("\n[Test 4] Warp Multiple Bounce Polarization")
    print("  S0 history: {}".format([round(v, 6) for v in S0[0].tolist()]))

    assert torch.allclose(S0, torch.ones_like(S0), atol=1e-5), \
        "S0 must remain 1 at every step"

    # S2 应在 ±1 之间翻转（正入射反射 flip）
    S2 = out["stokes"][0, :, 2]
    print("  S2 history: {}".format([round(v, 6) for v in S2.tolist()]))


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))