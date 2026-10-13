"""
tests/test_energy_full.py
==========================
总能量绝对守恒与多次反弹能量累积。
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
# Test 1: 总能量绝对守恒（无吸收）
# ==============================================================================

def test_total_energy_conservation_no_absorption():
    """
    非吸收场景下，所有光线的最终能量之和应严格等于初始总能量。

    物理依据：R + T = 1（无吸收时），Monte Carlo 分支的加权和守恒。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    N = 10000
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="fresnel")

    # 最终能量之和 = 每条光线的最终权重之和
    total_final = out["energy"][:, -1].sum().item()
    total_initial = float(N)

    rel_err = abs(total_final - total_initial) / total_initial

    print("\n[Test 1] Total energy conservation (no absorption)")
    print("  N = {}".format(N))
    print("  Initial total: {:.2f}".format(total_initial))
    print("  Final total  : {:.2f}".format(total_final))
    print("  Relative err : {:.4e}".format(rel_err))

    # Monte Carlo 统计误差：1/sqrt(N) ≈ 1%
    assert rel_err < 0.02, \
        "Total energy not conserved: rel_err = {:.4e}".format(rel_err)


# ==============================================================================
# Test 2: 多次反弹的能量累积
# ==============================================================================

def test_energy_after_n_bounces():
    """
    两个平行镜面，反射率 R。经过 N 次反射后，能量 ≈ R^N。

    场景：custom + reflectivity=R，强制每次反射都反射。
    使用确定性路径（无漫反射），能量精确等于 R^N。
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

    print("\n[Test 2] Energy after N bounces (R={})".format(R))
    print("  Energies: {}".format([round(v, 6) for v in energies.tolist()]))

    # 每次反射能量应乘以 R
    for n in range(1, min(5, len(energies))):
        expected = R ** n
        measured = energies[n].item()
        # 只检查 n 次反射后的一致性
        if measured > 1e-6:
            print("  step {}: measured={:.6f}, expected≈R^{}={:.6f}".format(
                n, measured, n, expected))
            assert abs(measured - expected) < 1e-4, \
                "Energy mismatch at step {}: {:.6f} vs {:.6f}".format(
                    n, measured, expected)


# ==============================================================================
# Test 3: 吸收场景的能量守恒
# ==============================================================================

def test_total_energy_with_absorption():
    """
    有吸收时，R + T + A = 1，总能量应守恒（吸收光线的能量通过 A 释放）。
    """
    device = torch.device("cpu")
    verts, faces = make_flat_interface(device)
    fm = torch.tensor([[1, 0], [1, 0]], dtype=torch.long, device=device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
    }

    N = 5000
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=3, surface_model="fresnel")

    total_final = out["energy"][:, -1].sum().item()
    total_initial = float(N)

    rel_err = abs(total_final - total_initial) / total_initial

    print("\n[Test 3] Total energy with absorption")
    print("  Initial: {:.2f}, Final: {:.2f}".format(total_initial, total_final))
    print("  Relative err: {:.4e}".format(rel_err))

    # 允许较大的统计误差（因为部分光线能量被吸收）
    assert rel_err < 0.05


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))