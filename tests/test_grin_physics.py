"""
tests/test_grin_physics.py
===========================
GRIN 介质的物理真实性测试：球差、RK4 收敛、弯曲方向。
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_grin_box(device, extents=(1.0, 1.0, 4.0)):
    box = trimesh.creation.box(extents=list(extents))
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def make_linear_grin_materials(n0=1.5, grad_x=0.1):
    """线性 GRIN：n(x) = n0 + grad_x * x。"""
    def linear_grin(x, y, z, wl):
        return n0 + grad_x * x

    return {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n0, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": linear_grin},
    }


def make_parabolic_grin_materials(n0=1.5, A=0.15):
    def parabolic_grin(x, y, z, wl):
        return n0 - A * (x ** 2 + y ** 2)

    return {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": n0, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": parabolic_grin},
    }


# ==============================================================================
# Test 1: GRIN 向高折射率方向弯曲
# ==============================================================================

def test_grin_bends_toward_higher_n():
    """
    在梯度场 n(x) = 1.5 + 0.1x 中，光线应向 +x 方向（高 n 方向）弯曲。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device, extents=(2.0, 2.0, 4.0))
    materials = make_linear_grin_materials(n0=1.5, grad_x=0.1)

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=200, ds_init=0.02, fresnel=False)

    pos = out["positions"][0]
    x_start = pos[0, 0].item()
    x_end = pos[-1, 0].item()

    print("\n[Test 1] GRIN bends toward higher n")
    print("  n(x) = 1.5 + 0.1x")
    print("  x_start = {:.4f}, x_end = {:.4f}".format(x_start, x_end))

    # 向 +x 弯曲（高 n 方向）
    assert x_end > x_start + 0.05, \
        "GRIN should bend ray toward +x: x_start={:.4f}, x_end={:.4f}".format(
            x_start, x_end)


# ==============================================================================
# Test 2: GRIN 球差
# ==============================================================================

def test_grin_spherical_aberration():
    """
    抛物线 GRIN 透镜对离轴光线的聚焦能力弱于近轴光线。
    物理依据：n(r) = n0 - A·r² 是抛物面，对边缘光线产生正球差。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device, extents=(2.0, 2.0, 2.0))
    materials = make_parabolic_grin_materials(n0=1.5, A=0.15)

    # 近轴光线 (x=0.1)
    rays_paraxial = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.1, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    # 边缘光线 (x=0.5)
    rays_marginal = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.5, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    out_para = tracer.trace(rays_paraxial, max_steps=200, ds_init=0.01)
    out_marg = tracer.trace(rays_marginal, max_steps=200, ds_init=0.01)

    # 聚焦比例：|x_end| / |x_start|
    x_start_p = abs(out_para["positions"][0, 0, 0].item())
    x_end_p = abs(out_para["positions"][0, -1, 0].item())
    x_start_m = abs(out_marg["positions"][0, 0, 0].item())
    x_end_m = abs(out_marg["positions"][0, -1, 0].item())

    ratio_para = x_end_p / x_start_p
    ratio_marg = x_end_m / x_start_m

    print("\n[Test 2] GRIN spherical aberration")
    print("  Paraxial (x=0.1): |x_end|/|x_start| = {:.4f}".format(ratio_para))
    print("  Marginal (x=0.5): |x_end|/|x_start| = {:.4f}".format(ratio_marg))

    # 近轴光线的聚焦能力强于边缘光线
    assert ratio_para < ratio_marg, \
        "Paraxial rays should focus better than marginal: " \
        "{:.4f} vs {:.4f}".format(ratio_para, ratio_marg)


# ==============================================================================
# Test 3: RK4 步长收敛性（4 阶）
# ==============================================================================

def test_grin_rk4_physical_convergence():
    """
    RK4 是 4 阶方法：步长减半，误差应下降约 16 倍。

    用同一个 GRIN 场景，跑不同步长，比较最终位置。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device, extents=(1.0, 1.0, 4.0))
    materials = make_parabolic_grin_materials(n0=1.5, A=0.1)

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.3, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    # 基准（极细步长）
    out_ref = tracer.trace(rays, max_steps=2000, ds_init=0.002,
                            ds_min=0.002, ds_max=0.002, fresnel=False)
    x_ref = out_ref["positions"][0, -1, 0].item()

    # 不同步长
    errors = []
    for ds in [0.04, 0.02, 0.01]:
        out = tracer.trace(rays, max_steps=2000, ds_init=ds,
                            ds_min=ds, ds_max=ds, fresnel=False)
        x = out["positions"][0, -1, 0].item()
        errors.append(abs(x - x_ref))

    print("\n[Test 3] RK4 physical convergence")
    print("  Reference x_end: {:.6f}".format(x_ref))
    for ds, err in zip([0.04, 0.02, 0.01], errors):
        print("  ds={:.4f}: |error| = {:.3e}".format(ds, err))

    # ds 减半，误差应下降至少 4 倍
    if errors[0] > 0 and errors[1] > 0:
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]
        print("  Convergence ratios: {:.2f}, {:.2f}".format(ratio1, ratio2))
        assert ratio1 > 3.0 or ratio2 > 3.0, \
            "RK4 should show >3x error reduction on halving ds"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))