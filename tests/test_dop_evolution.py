"""
tests/test_dop_evolution.py
============================
DoP（偏振度）经过多次界面的演化验证。

物理依据：
    Jones 矩阵 diag(r_s, r_p) 作用于 Stokes 向量：
        S0' = a·S0 + b·S1
        S1' = b·S0 + a·S1
        S2' = c·S2
        S3' = c·S3
    其中 a = (r_s² + r_p²)/2, b = (r_s² − r_p²)/2, c = r_s·r_p

    DoP² = (S1'² + S2'² + S3'²) / S0'²
"""
import math
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


# ==============================================================================
# 场景构造
# ==============================================================================

def make_two_mirrors(device, hs=5.0):
    """两个平行界面 z=0 和 z=1。"""
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
        [-hs, -hs, 1.0], [hs, -hs, 1.0], [hs, hs, 1.0], [-hs, hs, 1.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)
    fm = torch.zeros((4, 2), dtype=torch.long, device=device)
    fm[0:2, 0] = 0
    fm[0:2, 1] = 1
    fm[2:4, 0] = 1
    fm[2:4, 1] = 0
    return verts, faces, fm


def make_flat_interface(device, hs=100.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


# ==============================================================================
# Test 1: DoP 始终在 [0, 1] 范围
# ==============================================================================

def test_dop_stays_in_range_over_multiple_interfaces():
    """
    部分偏振光经过 3 次界面，DoP 应始终保持 [0, 1]。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 部分偏振输入
    rays = generate_rays(
        source_type="grid", num_rays=8,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)
    # 手动改为部分偏振：S1=0.3, S2=0.3, S3=0.2
    rays["stokes"][:, 1] = 0.3
    rays["stokes"][:, 2] = 0.3
    rays["stokes"][:, 3] = 0.2

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=10, surface_model="fresnel")

    dop = out["dop"]

    print("\n[Test 1] DoP stays in [0, 1]")
    print("  DoP range: [{:.6f}, {:.6f}]".format(
        dop.min().item(), dop.max().item()))

    assert torch.all(dop >= -1e-6), "DoP < 0 detected"
    assert torch.all(dop <= 1.0 + 1e-6), "DoP > 1 detected"


# ==============================================================================
# Test 2: 纯偏振输入经无损界面保持 DoP = 1
# ==============================================================================

def test_pure_polarization_preserves_dop():
    """
    纯偏振输入 (DoP = 1) 经过反射界面，DoP 应保持为 1。
    物理依据：Fresnel 反射是酉变换（乘以实标量不引入退偏振）。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=8,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="fresnel")

    dop = out["dop"]

    print("\n[Test 2] Pure polarization preserves DoP=1")
    print("  DoP range: [{:.6f}, {:.6f}]".format(
        dop.min().item(), dop.max().item()))

    # 纯偏振输入的 DoP = 1，经酉变换后 DoP 仍 = 1
    assert torch.allclose(dop, torch.ones_like(dop), atol=1e-4), \
        "DoP should stay at 1 for pure polarization input"


# ==============================================================================
# Test 3: DoP 数值与解析 Mueller 公式一致
# ==============================================================================

def test_dop_matches_analytic_mueller():
    """
    部分偏振输入经单次界面后，DoP 应与解析 Mueller 公式一致。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    theta_i = math.radians(30.0)
    S1_in, S2_in, S3_in = 0.5, 0.3, 0.1

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0),
        direction=(math.sin(theta_i), 0.0, math.cos(theta_i)),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)
    rays["stokes"][0, 1] = S1_in
    rays["stokes"][0, 2] = S2_in
    rays["stokes"][0, 3] = S3_in

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    # 解析 Mueller
    sin_t = 1.0 * math.sin(theta_i) / 1.5
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    r_s = (1.0 * cos_i - 1.5 * cos_t) / (1.0 * cos_i + 1.5 * cos_t)
    r_p = (1.5 * cos_i - 1.0 * cos_t) / (1.5 * cos_i + 1.0 * cos_t)

    a = 0.5 * (r_s ** 2 + r_p ** 2)
    b = 0.5 * (r_s ** 2 - r_p ** 2)
    c = r_s * r_p

    S0_raw = a * 1.0 + b * S1_in
    S1_raw = b * 1.0 + a * S1_in
    S2_raw = c * S2_in
    S3_raw = c * S3_in

    S1_norm = S1_raw / S0_raw
    S2_norm = S2_raw / S0_raw
    S3_norm = S3_raw / S0_raw
    dop_expected = math.sqrt(S1_norm ** 2 + S2_norm ** 2 + S3_norm ** 2)

    dop_measured = out["dop"][0, 1, 0].item()

    print("\n[Test 3] DoP matches analytic Mueller")
    print("  Input : S1={:.4f}, S2={:.4f}, S3={:.4f}".format(
        S1_in, S2_in, S3_in))
    print("  Expected DoP: {:.6f}".format(dop_expected))
    print("  Measured DoP: {:.6f}".format(dop_measured))

    assert abs(dop_measured - dop_expected) < 1e-4, \
        "DoP mismatch: {:.6f} vs {:.6f}".format(dop_measured, dop_expected)


# ==============================================================================
# Test 4: 多次反弹的 DoP 累积
# ==============================================================================

def test_dop_accumulates_over_bounces():
    """
    在两个平行镜面之间多次反弹，DoP 应保持不变（纯偏振）或按 Mueller 累积。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_two_mirrors(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=6, surface_model="custom")

    dop_history = out["dop"][0, :, 0]

    print("\n[Test 4] DoP accumulates over bounces")
    print("  DoP history: {}".format(
        [round(v, 6) for v in dop_history.tolist()]))

    # 纯偏振（45°）经多次反射：DoP 保持 1
    assert torch.allclose(dop_history, torch.ones_like(dop_history), atol=1e-4)


# ==============================================================================
# Test 5: 吸收界面导致的 DoP 变化
# ==============================================================================

def test_dop_through_absorbing_interface():
    """
    有吸收的界面，DoP 应保持 [0, 1] 范围。

    物理：吸收率不区分 s/p，因此不引入退偏振（DoP 不变）。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
    }

    rays = generate_rays(
        source_type="grid", num_rays=8,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="fresnel")

    dop = out["dop"]

    print("\n[Test 5] DoP through absorbing interface")
    print("  DoP range: [{:.6f}, {:.6f}]".format(
        dop.min().item(), dop.max().item()))

    assert torch.all(dop >= -1e-6) and torch.all(dop <= 1.0 + 1e-6)


# ==============================================================================
# Test 6: 漫反射导致的 DoP → 0
# ==============================================================================

def test_diffuse_collapses_dop():
    """
    漫反射将偏振完全随机化，DoP → 0。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 1.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 1.0, "absorptivity": 0.0},
    }

    N = 256
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.01, 0.01, -0.01, 0.01),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    # step 1 之后 DoP 应塌缩到 0
    dop_step1 = out["dop"][:, 1, 0]

    print("\n[Test 6] Diffuse collapses DoP")
    print("  DoP after diffuse: max = {:.6f}".format(dop_step1.max().item()))

    assert torch.all(dop_step1 < 1e-4), \
        "DoP should collapse to 0 after diffuse reflection"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))