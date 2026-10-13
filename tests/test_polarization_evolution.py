"""
tests/test_polarization_evolution.py
=====================================
偏振随反射/折射演化的完整验证——补齐 5 个缺口：

    1. 纯 s / p 偏振经界面后保持不变（S1 = ±1 invariant）
    2. 偏振随入射角的完整曲线（与解析 Fresnel 公式连续一致）
    3. 多次透射的偏振累积（穿过玻璃板，多次 t_s/t_p 作用）
    4. 部分偏振（DoP < 1）经界面的精确解析 Mueller 公式
    5. Stokes 参考系旋转的往返一致性

Run:
    pytest tests/test_polarization_evolution.py -v -s
"""
import math
import numpy as np
import pytest
import torch

from georay import AdvancedRayTracerPBRTGPU, generate_rays
from georay.polarization import rotate_stokes_reference


# ==============================================================================
# 几何辅助
# ==============================================================================

def make_flat_interface(device, hs=100.0):
    """单平面 z=0，法线 +z。"""
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def make_slab_z(device, z0=0.0, z1=1.0, hs=100.0):
    """
    一个玻璃薄板：z ∈ [z0, z1] 是玻璃，其他是空气。
    用两个大平行平面表示（无侧面），用于测试多次透射。

    fm 布置：
        z=z0 平面（法线 +z）：fm[:,0]=1（玻璃，+z 侧），fm[:,1]=0（空气）
        z=z1 平面（法线 +z）：fm[:,0]=0（空气），fm[:,1]=1（玻璃）
    """
    verts = torch.tensor([
        [-hs, -hs, z0], [hs, -hs, z0], [hs, hs, z0], [-hs, hs, z0],
        [-hs, -hs, z1], [hs, -hs, z1], [hs, hs, z1], [-hs, hs, z1],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
    ], dtype=torch.long, device=device)

    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[0:2, 0] = 1
    fm[0:2, 1] = 0
    fm[2:4, 0] = 0
    fm[2:4, 1] = 1
    return verts, faces, fm


# ==============================================================================
# Fresnel 解析辅助
# ==============================================================================

def fresnel_rs_rp(n1, n2, theta_i):
    sin_t = n1 * math.sin(theta_i) / n2
    if sin_t >= 1.0:
        return 1.0, 1.0, 0.0
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return r_s, r_p, cos_t


def fresnel_ts_tp(n1, n2, theta_i):
    sin_t = n1 * math.sin(theta_i) / n2
    cos_t = math.sqrt(1.0 - sin_t ** 2)
    cos_i = math.cos(theta_i)
    t_s = 2.0 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
    t_p = 2.0 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)
    return t_s, t_p


# ==============================================================================
# 缺口 1：纯 s / p 偏振经界面后保持不变
# ==============================================================================

@pytest.mark.parametrize("pol,S1_in", [("s", 1.0), ("p", -1.0)])
def test_pure_polarization_invariant(pol, S1_in):
    """
    纯 s 偏振（S1 = +1）经界面后仍是纯 s；
    纯 p 偏振（S1 = -1）经界面后仍是纯 p。

    物理依据：Jones 矩阵 diag(r_s, r_p) 是对角的，s / p 分量不耦合。
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_i = math.radians(30.0)

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)

    # 反射测试
    materials_ref = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }
    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))
    rays_ref = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=pol, device=device)

    tracer_ref = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials_ref, device)
    out_ref = tracer_ref.trace(rays_ref, max_steps=3, surface_model="custom")

    S1_ref = out_ref["stokes"][0, 1, 1].item()
    S2_ref = out_ref["stokes"][0, 1, 2].item()
    S3_ref = out_ref["stokes"][0, 1, 3].item()

    print("\n[Gap 1] {} polarization invariant test".format(pol))
    print("  Input  S1 = {:+.6f}".format(S1_in))
    print("  After reflection : S1={:+.6f}, S2={:+.6f}, S3={:+.6f}".format(
        S1_ref, S2_ref, S3_ref))

    assert abs(S1_ref - S1_in) < 1e-5, \
        "{} after reflection: S1={}, expected {}".format(pol, S1_ref, S1_in)
    assert abs(S2_ref) < 1e-5, \
        "{} after reflection: S2={}, expected 0".format(pol, S2_ref)
    assert abs(S3_ref) < 1e-5, \
        "{} after reflection: S3={}, expected 0".format(pol, S3_ref)

    # 折射测试
    materials_tr = {
        0: {"n": n1, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }
    rays_tr = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=pol, device=device)

    tracer_tr = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials_tr, device)
    out_tr = tracer_tr.trace(rays_tr, max_steps=3, surface_model="custom")

    S1_tr = out_tr["stokes"][0, 1, 1].item()
    S2_tr = out_tr["stokes"][0, 1, 2].item()
    S3_tr = out_tr["stokes"][0, 1, 3].item()

    print("  After refraction : S1={:+.6f}, S2={:+.6f}, S3={:+.6f}".format(
        S1_tr, S2_tr, S3_tr))

    assert abs(S1_tr - S1_in) < 1e-5, \
        "{} after refraction: S1={}, expected {}".format(pol, S1_tr, S1_in)
    assert abs(S2_tr) < 1e-5
    assert abs(S3_tr) < 1e-5

    # s / p 之间不会互相转换
    print("  -> {} polarization preserved through both reflection & refraction".format(pol))


# ==============================================================================
# 缺口 2：偏振随入射角的完整曲线
# ==============================================================================

def test_polarization_vs_angle_curve():
    """
    45° 线偏振输入，扫描 θi ∈ [5°, 85°]，比较反射后 S1, S2 与解析 Fresnel 公式。

    物理依据：
        S1_out(θi) = (R_s − R_p) / (R_s + R_p)
        S2_out(θi) = 2 · r_s · r_p / (R_s + R_p)

    曲线应该连续并与解析解一致，包括 Brewster 角附近 r_p 变号的拐点。
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    angles_deg = np.linspace(5.0, 85.0, 17)

    print("\n[Gap 2] Polarization vs Angle Curve (reflection, 45-deg linear input)")
    print("  {:>8s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
        "theta", "S1_expected", "S1_measured", "S2_expected", "S2_measured"))
    print("  " + "-" * 70)

    max_err_s1 = 0.0
    max_err_s2 = 0.0

    for theta_deg in angles_deg:
        theta = math.radians(theta_deg)
        r_s, r_p, _ = fresnel_rs_rp(n1, n2, theta)
        R_s, R_p = r_s ** 2, r_p ** 2
        S1_expected = (R_s - R_p) / (R_s + R_p)
        S2_expected = 2.0 * r_s * r_p / (R_s + R_p)

        direction = (math.sin(theta), 0.0, math.cos(theta))
        rays = generate_rays(
            source_type="grid", num_rays=1,
            origin=(0.0, 0.0, -1.0), direction=direction,
            bounds=(0.0, 0.0, 0.0, 0.0),
            polarization=45.0, device=device)

        out = tracer.trace(rays, max_steps=3, surface_model="custom")
        S1_meas = out["stokes"][0, 1, 1].item()
        S2_meas = out["stokes"][0, 1, 2].item()

        err_s1 = abs(S1_meas - S1_expected)
        err_s2 = abs(S2_meas - S2_expected)
        max_err_s1 = max(max_err_s1, err_s1)
        max_err_s2 = max(max_err_s2, err_s2)

        print("  {:>8.1f} {:>14.6f} {:>14.6f} {:>14.6f} {:>14.6f}".format(
            theta_deg, S1_expected, S1_meas, S2_expected, S2_meas))

    print("  " + "-" * 70)
    print("  Max |S1 error| = {:.2e}".format(max_err_s1))
    print("  Max |S2 error| = {:.2e}".format(max_err_s2))

    assert max_err_s1 < 1e-4, \
        "S1 vs angle curve mismatch: max |delta| = {:.6e}".format(max_err_s1)
    assert max_err_s2 < 1e-4, \
        "S2 vs angle curve mismatch: max |delta| = {:.6e}".format(max_err_s2)

    # 检查曲线连续性——Brewster 角附近 S2 应变号
    theta_B = math.atan(n2 / n1)
    print("  Brewster angle = {:.2f} deg".format(math.degrees(theta_B)))


# ==============================================================================
# 缺口 3：多次透射的偏振累积
# ==============================================================================

def test_polarization_multiple_refractions():
    """
    光线穿过玻璃薄板（2 次透射：空气→玻璃，玻璃→空气），
    验证累积偏振与解析 Jones 矩阵级联一致。

    物理：
        E_s_out = t_s(air->glass) * t_s(glass->air) * E_s_in
        E_p_out = t_p(air->glass) * t_p(glass->air) * E_p_in
        S1_out = (|E_s_out|^2 - |E_p_out|^2) / (|E_s_out|^2 + |E_p_out|^2)
    """
    device = torch.device("cpu")
    n_air, n_glass = 1.0, 1.5
    theta_i = math.radians(30.0)

    verts, faces, fm = make_slab_z(device, z0=0.0, z1=1.0, hs=100.0)

    # custom + reflectivity=0 → 纯透射
    materials = {
        0: {"n": n_air, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n_glass, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(math.sin(theta_i), 0.0, math.cos(theta_i)),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization=45.0, device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=6, surface_model="custom")

    # 解析：
    #   第一次：空气 → 玻璃 (θi = 30°, θt = asin(sin30/1.5) = 19.47°)
    #   第二次：玻璃 → 空气 (θi = 19.47°, θt = 30°)
    t_s_ag, t_p_ag = fresnel_ts_tp(n_air, n_glass, theta_i)
    theta_t = math.asin(n_air * math.sin(theta_i) / n_glass)
    t_s_ga, t_p_ga = fresnel_ts_tp(n_glass, n_air, theta_t)

    E_s_total = t_s_ag * t_s_ga
    E_p_total = t_p_ag * t_p_ga

    I_s = E_s_total ** 2
    I_p = E_p_total ** 2
    S1_expected = (I_s - I_p) / (I_s + I_p)

    # 从 tracer 输出读取
    S1_step1 = out["stokes"][0, 1, 1].item()  # 第 1 次透射后
    S1_step2 = out["stokes"][0, 2, 1].item()  # 第 2 次透射后（穿出玻璃）

    # 单次透射的解析
    I_s1 = t_s_ag ** 2
    I_p1 = t_p_ag ** 2
    S1_expected_1 = (I_s1 - I_p1) / (I_s1 + I_p1)

    print("\n[Gap 3] Polarization Multiple Refractions (single slab, 2 transmissions)")
    print("  theta_i (air) = {:.2f} deg".format(math.degrees(theta_i)))
    print("  theta_t (glass) = {:.2f} deg".format(math.degrees(theta_t)))
    print()
    print("  t_s(air->glass) = {:.6f}, t_p(air->glass) = {:.6f}".format(
        t_s_ag, t_p_ag))
    print("  t_s(glass->air) = {:.6f}, t_p(glass->air) = {:.6f}".format(
        t_s_ga, t_p_ga))
    print()
    print("  After 1st transmission: S1_expected={:+.6f}, S1_measured={:+.6f}".format(
        S1_expected_1, S1_step1))
    print("  After 2nd transmission: S1_expected={:+.6f}, S1_measured={:+.6f}".format(
        S1_expected, S1_step2))

    assert abs(S1_step1 - S1_expected_1) < 1e-4, \
        "S1 after 1st refraction: got {}, expected {}".format(
            S1_step1, S1_expected_1)
    assert abs(S1_step2 - S1_expected) < 1e-4, \
        "S1 after 2nd refraction: got {}, expected {}".format(
            S1_step2, S1_expected)

    # 偏振随累积透射递增（因为 t_p > t_s 恒成立，偏振趋向 -1）
    assert abs(S1_step2) > abs(S1_step1), \
        "Polarization should accumulate across multiple refractions"
    print("  -> Polarization accumulates: |S1| grows with number of transmissions")


# ==============================================================================
# 缺口 4：部分偏振经界面的精确解析 Mueller 公式
# ==============================================================================

def test_partial_polarization_mueller_formula():
    """
    部分偏振（DoP < 1）经过 Fresnel 界面的精确解析 Mueller 公式：

        S0' = a·S0 + b·S1
        S1' = b·S0 + a·S1
        S2' = c·S2
        S3' = c·S3

    其中 a = 0.5(r_s^2 + r_p^2)，b = 0.5(r_s^2 - r_p^2)，c = r_s·r_p。

    输出归一化 Stokes 为 (1, S1'/S0', S2'/S0', S3'/S0')。
    """
    device = torch.device("cpu")
    n1, n2 = 1.0, 1.5
    theta_i = math.radians(30.0)

    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": n1, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n2, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # 输入部分偏振：S1=0.5, S2=0.3, S3=0.1 → DoP = sqrt(0.35) ≈ 0.592
    S1_in, S2_in, S3_in = 0.5, 0.3, 0.1
    dop_in = math.sqrt(S1_in ** 2 + S2_in ** 2 + S3_in ** 2)

    direction = (math.sin(theta_i), 0.0, math.cos(theta_i))
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=direction,
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    # 覆盖为部分偏振态
    rays["stokes"][0, 0] = 1.0
    rays["stokes"][0, 1] = S1_in
    rays["stokes"][0, 2] = S2_in
    rays["stokes"][0, 3] = S3_in

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="custom")

    # 解析 Mueller 公式
    r_s, r_p, _ = fresnel_rs_rp(n1, n2, theta_i)
    a = 0.5 * (r_s * r_s + r_p * r_p)
    b = 0.5 * (r_s * r_s - r_p * r_p)
    c = r_s * r_p

    S0_out_raw = a * 1.0 + b * S1_in
    S1_out_raw = b * 1.0 + a * S1_in
    S2_out_raw = c * S2_in
    S3_out_raw = c * S3_in

    S1_out_expected = S1_out_raw / S0_out_raw
    S2_out_expected = S2_out_raw / S0_out_raw
    S3_out_expected = S3_out_raw / S0_out_raw
    dop_out_expected = math.sqrt(
        S1_out_expected ** 2 + S2_out_expected ** 2 + S3_out_expected ** 2)

    S1_out = out["stokes"][0, 1, 1].item()
    S2_out = out["stokes"][0, 1, 2].item()
    S3_out = out["stokes"][0, 1, 3].item()
    dop_out = math.sqrt(S1_out ** 2 + S2_out ** 2 + S3_out ** 3 if False else S1_out ** 2 + S2_out ** 2 + S3_out ** 2)

    print("\n[Gap 4] Partial Polarization Mueller Formula")
    print("  theta_i = 30 deg")
    print("  Input  : S1={:+.4f}, S2={:+.4f}, S3={:+.4f}, DoP={:.4f}".format(
        S1_in, S2_in, S3_in, dop_in))
    print("  Expected output: S1={:+.6f}, S2={:+.6f}, S3={:+.6f}, DoP={:.6f}".format(
        S1_out_expected, S2_out_expected, S3_out_expected, dop_out_expected))
    print("  Measured output: S1={:+.6f}, S2={:+.6f}, S3={:+.6f}, DoP={:.6f}".format(
        S1_out, S2_out, S3_out, dop_out))

    assert abs(S1_out - S1_out_expected) < 1e-4, \
        "S1 mismatch: got {}, expected {}".format(S1_out, S1_out_expected)
    assert abs(S2_out - S2_out_expected) < 1e-4, \
        "S2 mismatch: got {}, expected {}".format(S2_out, S2_out_expected)
    assert abs(S3_out - S3_out_expected) < 1e-4, \
        "S3 mismatch: got {}, expected {}".format(S3_out, S3_out_expected)

    # 注意：DoP 不保持——Fresnel 界面对 s / p 分量作用不同，
    # 会导致部分偏振光的 DoP 变化。这是物理上正确的。
    print("  -> DoP changed: {} -> {} (expected, Fresnel interface is s/p-selective)".format(
        round(dop_in, 4), round(dop_out, 4)))


# ==============================================================================
# 缺口 5：Stokes 参考系旋转的往返一致性
# ==============================================================================

def test_stokes_reference_rotation_roundtrip():
    """
    rotate_stokes_reference(S, s_old, s_new, d) 的往返一致性：

        S' = rotate(S, s_old, s_new, d)
        S'' = rotate(S', s_new, s_old, d)
        S'' == S

    这个不变量在 tracer 内部被隐含使用，此处直接验证。
    """
    device = torch.device("cpu")

    # 场景 1：正交 s 向量（90° 旋转）
    d1 = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    s1 = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    s2 = torch.tensor([[0.0, 1.0, 0.0]], device=device)

    stokes = torch.tensor([[1.0, 0.5, -0.3, 0.2]], device=device)

    rotated = rotate_stokes_reference(stokes, s1, s2, d1)
    recovered = rotate_stokes_reference(rotated, s2, s1, d1)

    err1 = (stokes - recovered).abs().max().item()
    print("\n[Gap 5] Stokes Reference Rotation Roundtrip")
    print("  Case 1 (orthogonal s vectors, 90-deg rotation):")
    print("    Original   : {}".format(stokes.tolist()))
    print("    Rotated    : {}".format(rotated.tolist()))
    print("    Recovered  : {}".format(recovered.tolist()))
    print("    Max error  : {:.2e}".format(err1))

    assert err1 < 1e-6, \
        "Roundtrip (orthogonal) failed: max |S - S''| = {}".format(err1)

    # 场景 2：非正交 s 向量（45° 旋转）
    s3 = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    s4 = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5), 0.0]], device=device)

    rotated2 = rotate_stokes_reference(stokes, s3, s4, d1)
    recovered2 = rotate_stokes_reference(rotated2, s4, s3, d1)

    err2 = (stokes - recovered2).abs().max().item()
    print("  Case 2 (non-orthogonal s vectors, 45-deg rotation):")
    print("    Original   : {}".format(stokes.tolist()))
    print("    Rotated    : {}".format(rotated2.tolist()))
    print("    Recovered  : {}".format(recovered2.tolist()))
    print("    Max error  : {:.2e}".format(err2))

    assert err2 < 1e-6, \
        "Roundtrip (non-orthogonal) failed: max |S - S''| = {}".format(err2)

    # 场景 3：沿斜方向的 d
    d3 = torch.tensor([[0.6, 0.0, 0.8]], device=device)
    s5 = torch.tensor([[0.8, 0.0, -0.6]], device=device)  # 与 d3 正交
    s6 = torch.tensor([[0.0, 1.0, 0.0]], device=device)   # 与 d3 正交

    rotated3 = rotate_stokes_reference(stokes, s5, s6, d3)
    recovered3 = rotate_stokes_reference(rotated3, s6, s5, d3)

    err3 = (stokes - recovered3).abs().max().item()
    print("  Case 3 (oblique d):")
    print("    Max error  : {:.2e}".format(err3))

    assert err3 < 1e-6, \
        "Roundtrip (oblique d) failed: max |S - S''| = {}".format(err3)

    # 场景 4：S3 分量应保持不变（旋转只影响 S1/S2）
    s7 = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    s8 = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    rotated4 = rotate_stokes_reference(stokes, s7, s8, d1)
    assert abs(rotated4[0, 3].item() - stokes[0, 3].item()) < 1e-6, \
        "S3 must be invariant under reference rotation"
    assert abs(rotated4[0, 0].item() - stokes[0, 0].item()) < 1e-6, \
        "S0 must be invariant under reference rotation"

    print("  Case 4: S0 and S3 invariant under rotation: OK")
    print("  -> All rotation roundtrip tests passed")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))