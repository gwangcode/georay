"""
tests/test_energy_full.py
==========================
总能量与路径权重的正确物理验证。

物理背景：
    Monte Carlo 单路径追踪中，每条光线只走一个分支。
    因此单条光线的权重不守恒（是抽样结果）。
    但以下情况是严格守恒的：
      * 全反射场景（R=1）：每条光线权重恒为 1
      * 全透射场景（T=1）：每条光线权重恒为 1
      * 界面 R + T + A = 1（每个界面的概率归一化）
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_flat_interface(device, hs=100.0):
    verts = torch.tensor([
        [-hs, -hs, 0.0], [hs, -hs, 0.0], [hs, hs, 0.0], [-hs, hs, 0.0],
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    return verts, faces


def make_two_mirrors(device, hs=5.0):
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


# ==============================================================================
# Test 1: 全反射场景下权重恒为 1
# ==============================================================================

def test_weight_unity_pure_reflection():
    """
    纯反射场景（custom + reflectivity=1）：每条光线权重恒为 1。

    ★ 修复：N 必须用完全平方数（grid 光源会取整），
       或者直接从 ray_dict 读实际光线数。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # ★ 用完全平方数避免 grid 光源被截断
    N_request = 1024   # 32²
    rays = generate_rays(
        source_type="grid", num_rays=N_request,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    # ★ 用实际光线数作为初始总能量
    N = rays["origins"].shape[0]

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="custom")

    total_final = out["energy"][:, -1].sum().item()
    total_initial = float(N)

    rel_err = abs(total_final - total_initial) / total_initial

    print("\n[Test 1] Weight unity (pure reflection)")
    print("  Requested N: {}".format(N_request))
    print("  Actual N   : {}".format(N))
    print("  Initial total: {:.2f}".format(total_initial))
    print("  Final total  : {:.2f}".format(total_final))
    print("  Relative err : {:.4e}".format(rel_err))

    assert rel_err < 1e-5, \
        "Pure reflection should preserve total weight: rel_err = {:.4e}".format(rel_err)

# ==============================================================================
# Test 2: 分支权重是 R 或 1-R
# ==============================================================================

def test_energy_ratio_is_branch_probability():
    """
    custom 模式 + reflectivity=R 时，每次反射/折射：
        权重比值 ∈ {P_specular, P_refract} = {R, 1-R}

    ★ 修复：原测试断言能量 = R^n，但 Monte Carlo 每次只走一个分支，
       能量是 branch_prob 的累乘。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_two_mirrors(device, hs=5.0)
    R = 0.8
    materials = {
        0: {"n": 1.0, "reflectivity": R, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": R, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, 0.5), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=6, surface_model="custom")

    energies = out["energy"][0]

    print("\n[Test 2] Energy ratio = branch probability (R={})".format(R))
    print("  Energies: {}".format([round(v, 6) for v in energies.tolist()]))

    # 检查相邻步的比值 ∈ {R, 1-R}
    for n in range(1, len(energies)):
        prev = energies[n-1].item()
        cur = energies[n].item()
        if prev > 1e-8:
            ratio = cur / prev
            ok = (abs(ratio - R) < 1e-4 or
                  abs(ratio - (1 - R)) < 1e-4 or
                  abs(ratio - 1.0) < 1e-6)
            print("  step {}: ratio = {:.6f} (R={}, 1-R={})".format(
                n, ratio, R, 1 - R))
            assert ok, \
                "Step {}: ratio {:.6f} not in {{R, 1-R, 1}}".format(n, ratio)

    assert out["energy_conserved"]


# ==============================================================================
# Test 3: 界面 R + T + A = 1
# ==============================================================================

def test_interface_probability_normalized():
    """
    每个界面上 R + T + A = 1（概率归一化）。
    这是 Monte Carlo 追踪的正确性基础。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
    }

    N = 500
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    R = out["reflectivity"][:, 1, 0]
    T = out["transmissivity"][:, 1, 0]
    A = out["absorptivity"][:, 1, 0]
    total = R + T + A

    # 只检查实际发生了界面交互的光线
    interacted = total > 1e-6
    n_events = interacted.sum().item()

    print("\n[Test 3] Interface R+T+A = 1")
    print("  Interface events: {}".format(n_events))
    if n_events > 0:
        max_dev = (total[interacted] - 1.0).abs().max().item()
        print("  max |R+T+A-1| = {:.3e}".format(max_dev))
        assert max_dev < 1e-4

    assert out["energy_conserved"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))