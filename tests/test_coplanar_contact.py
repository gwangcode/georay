"""
tests/test_coplanar_contact.py
================================
两个物体接触共面场景的测试。

物理背景：
    当两个网格几何接触（共享一个面）时，共享面处会存在两个
    共面三角形。射线追踪器必须：
      1. 不要无限循环地来回弹跳
      2. 正确处理 prev_face_ids
      3. 保持能量守恒
      4. 不产生 NaN

Run:
    pytest tests/test_coplanar_contact.py -v
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


# ==============================================================================
# 场景构造辅助函数
# ==============================================================================

def make_two_cubes_sharing_face_x(device):
    """
    两个 1×2×2 的玻璃立方体在 x=0 处接触：
        Cube A: x ∈ [-1, 0]
        Cube B: x ∈ [ 0, 1]
    共享面在 x = 0。
    """
    a = trimesh.creation.box(extents=[1.0, 2.0, 2.0])
    a.apply_translation([-0.5, 0.0, 0.0])
    b = trimesh.creation.box(extents=[1.0, 2.0, 2.0])
    b.apply_translation([0.5, 0.0, 0.0])
    combined = trimesh.util.concatenate([a, b])

    verts = torch.tensor(combined.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0   # 外部 = 空气
    fm[:, 1] = 1   # 内部 = 玻璃
    return verts, faces, fm


def make_two_cubes_stacked_z(device):
    """
    两个 2×2×1 的玻璃立方体在 z=0 处堆叠：
        Bottom: z ∈ [-1, 0]
        Top:    z ∈ [ 0, 1]
    共享面在 z = 0。
    """
    bottom = trimesh.creation.box(extents=[2.0, 2.0, 1.0])
    bottom.apply_translation([0.0, 0.0, -0.5])
    top = trimesh.creation.box(extents=[2.0, 2.0, 1.0])
    top.apply_translation([0.0, 0.0, 0.5])
    combined = trimesh.util.concatenate([bottom, top])

    verts = torch.tensor(combined.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def make_rays_along_x(device, n=64):
    """沿 +x 方向的光线，穿过 x=0 处接触面。"""
    return generate_rays(
        source_type="grid", num_rays=n,
        origin=(-3.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        bounds=(-0.3, 0.3, -0.3, 0.3),
        polarization="s", device=device)


def make_rays_along_z(device, n=64):
    """沿 +z 方向的光线，穿过 z=0 处接触面。"""
    return generate_rays(
        source_type="grid", num_rays=n,
        origin=(0.0, 0.0, -3.0),
        direction=(0.0, 0.0, 1.0),
        bounds=(-0.3, 0.3, -0.3, 0.3),
        polarization="s", device=device)


# ==============================================================================
# Test 1: 两个立方体沿 x 方向接触
# ==============================================================================

def test_two_cubes_sharing_face_x():
    """
    两个相同玻璃立方体接触，共享 x=0 处的面。

    断言：
      - 无 NaN
      - 所有光线在 max_steps 内终止（无死循环）
      - 能量守恒不变量成立
      - 打印每条光线的终止步数与最终能量（诊断）
    """
    device = torch.device("cpu")
    verts, faces, fm = make_two_cubes_sharing_face_x(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = make_rays_along_x(device, n=64)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=20, surface_model="custom")

    # ---- 无 NaN ----
    assert not torch.isnan(out["positions"]).any(), "positions contains NaN"
    assert not torch.isnan(out["energy"]).any(), "energy contains NaN"

    # ---- 所有光线终止 ----
    final_active = out["active"][:, -1]
    n_stuck = final_active.sum().item()
    assert n_stuck == 0, \
        f"{n_stuck}/{rays['origins'].shape[0]} rays stuck at coplanar contact"

    # ---- 能量守恒 ----
    assert out["energy_conserved"], "energy_conserved flag is False"

    # ---- 诊断输出 ----
    pos = out["positions"]
    energy = out["energy"]
    term_steps = []
    for n in range(pos.shape[0]):
        for s in range(pos.shape[1]):
            if not out["active"][n, s]:
                term_steps.append(s)
                break
    if term_steps:
        print(f"\n[test 1] Termination step distribution: "
              f"min={min(term_steps)}, max={max(term_steps)}, "
              f"mean={sum(term_steps)/len(term_steps):.2f}")
    print(f"[test 1] Final energy range: "
          f"[{energy[:, -1].min().item():.4f}, "
          f"{energy[:, -1].max().item():.4f}]")


# ==============================================================================
# Test 2: 两个立方体沿 z 方向堆叠
# ==============================================================================

def test_two_cubes_stacked_z():
    """
    两个相同玻璃立方体垂直堆叠，共享 z=0 处的面。

    断言：
      - 无 NaN
      - 所有光线在 max_steps 内终止
      - 能量守恒不变量成立
    """
    device = torch.device("cpu")
    verts, faces, fm = make_two_cubes_stacked_z(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = make_rays_along_z(device, n=64)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=20, surface_model="custom")

    assert not torch.isnan(out["positions"]).any(), "positions contains NaN"

    final_active = out["active"][:, -1]
    n_stuck = final_active.sum().item()
    assert n_stuck == 0, \
        f"{n_stuck}/{rays['origins'].shape[0]} rays stuck at coplanar contact"

    assert out["energy_conserved"], "energy_conserved flag is False"

    pos = out["positions"]
    term_steps = []
    for n in range(pos.shape[0]):
        for s in range(pos.shape[1]):
            if not out["active"][n, s]:
                term_steps.append(s)
                break
    if term_steps:
        print(f"\n[test 2] Termination step distribution: "
              f"min={min(term_steps)}, max={max(term_steps)}, "
              f"mean={sum(term_steps)/len(term_steps):.2f}")


# ==============================================================================
# Test 3: 共面接触处的界面能量守恒 R + T + A = 1
# ==============================================================================

def test_coplanar_energy_conservation():
    """
    每次界面交互都必须满足 R + T + A ≈ 1，
    即使两个物体在共面边界处接触。

    使用非零吸收率来让 R、T、A 都非零，
    这样能真正检验三者之和是否严格守恒。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_two_cubes_sharing_face_x(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
    }

    rays = make_rays_along_x(device, n=32)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

    R = out["reflectivity"]   # [N, S+1, 1]
    T = out["transmissivity"]
    A = out["absorptivity"]

    total = R + T + A
    interacted = total > 1e-6

    if interacted.any():
        deviations = (total - 1.0)[interacted].abs()
        max_dev = deviations.max().item()
        print(f"\n[test 3] Max |R+T+A-1| across all interfaces: {max_dev:.3e}")
        print(f"[test 3] Mean |R+T+A-1|: {deviations.mean().item():.3e}")
        print(f"[test 3] Number of interface events: {interacted.sum().item()}")
        assert max_dev < 1e-4, \
            f"Energy conservation violated: max |R+T+A-1| = {max_dev:.6e}"
    else:
        pytest.fail("No interface events detected — scene may be broken")

    assert out["energy_conserved"], "energy_conserved flag is False"


# ==============================================================================
# Test 4: prev_face 处理——无重复交互 / 无死循环
# ==============================================================================

def test_coplanar_prev_face_handling():
    """
    prev_face_ids 机制必须正确处理共面接触。

    关键设计：使用极小的 max_steps=5。
    如果 prev_face 排除逻辑有问题，光线会卡在共面接触面上
    来回弹跳，无法在 5 步内逃逸。这是一个强有力的一致性检测。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_two_cubes_sharing_face_x(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = make_rays_along_x(device, n=32)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=5, surface_model="fresnel")

    # ---- 所有光线必须在 5 步内逃逸 ----
    final_active = out["active"][:, -1]
    n_stuck = final_active.sum().item()
    print(f"\n[test 4] {n_stuck}/{rays['origins'].shape[0]} rays "
          f"did not escape within 5 steps")

    if n_stuck > 0:
        # 打印诊断信息
        pos = out["positions"]
        for n in range(pos.shape[0]):
            if final_active[n]:
                print(f"  STUCK ray {n}:")
                for s in range(pos.shape[1]):
                    print(f"    step {s}: pos = {pos[n, s].tolist()}, "
                          f"active = {out['active'][n, s].item()}")

    assert n_stuck == 0, \
        f"{n_stuck} rays stuck at coplanar contact — possible infinite loop"

    # ---- 每条光线的终止步数 ----
    pos = out["positions"]
    for n in range(min(5, pos.shape[0])):
        term_step = None
        for s in range(pos.shape[1]):
            if not out["active"][n, s]:
                term_step = s
                break
        if term_step is not None:
            print(f"  ray {n}: escaped at step {term_step}, "
                  f"final pos = {[round(v, 4) for v in pos[n, term_step].tolist()]}")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))