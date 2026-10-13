"""
tests/test_coplanar_multi_object.py
====================================
多个物体（N ≥ 3）共面接触场景的测试。

物理/数值挑战：
    当 N 个网格几何共享同一个共面边界时：
      * 一个平面上可能出现大量共面三角形
      * prev_face_ids 只排除一个面，可能不够
      * BVH 需要在同一位置处理大量共面三角形
      * 数值误差更容易累积，可能引发 Z-fighting
      * Warp 内核的 `mesh_query_ray` 返回哪个面取决于实现细节

覆盖场景：
    1. 3 个立方体沿 +x 排成一行
    2. 5 层玻璃板沿 +z 堆叠
    3. 3×3 立方体网格，射线穿过网格方向
    4. 异构介质共面（n=1.5 接触 n=1.7）
    5. 8 个立方体长链（stress test）
    6. 多界面严格能量守恒

Run:
    pytest tests/test_coplanar_multi_object.py -v -s
"""
import math
import pytest
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


# ==============================================================================
# 场景构造
# ==============================================================================

def make_chain_of_cubes_x(device, n_cubes=3, cube_size=1.0):
    """
    n_cubes 个玻璃立方体沿 +x 排成一行：
        cube i 中心 = ((i + 0.5) * cube_size, 0, 0)
        cube i 范围 = [(i) * cs, (i+1) * cs] × [±cs/2] × [±cs/2]
    相邻 cube 在 x = (i+1)*cs 处共享一个矩形面。
    """
    meshes = []
    for i in range(n_cubes):
        c = trimesh.creation.box(extents=[cube_size, cube_size, cube_size])
        c.apply_translation([(i + 0.5) * cube_size, 0.0, 0.0])
        meshes.append(c)
    combined = trimesh.util.concatenate(meshes)

    verts = torch.tensor(combined.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0   # 外部 = 空气
    fm[:, 1] = 1   # 内部 = 玻璃
    return verts, faces, fm


def make_tower_of_slabs_z(device, n_layers=5, slab_thickness=0.2, slab_size=2.0):
    """
    n_layers 层玻璃板沿 +z 堆叠：
        slab i 中心 = (0, 0, (i + 0.5) * thickness)
    相邻板在 z = (i+1)*thickness 处共享一个面。
    """
    meshes = []
    for i in range(n_layers):
        s = trimesh.creation.box(extents=[slab_size, slab_size, slab_thickness])
        s.apply_translation([0.0, 0.0, (i + 0.5) * slab_thickness])
        meshes.append(s)
    combined = trimesh.util.concatenate(meshes)

    verts = torch.tensor(combined.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def make_grid_of_cubes_xy(device, nx=3, ny=3, cube_size=1.0):
    """
    nx × ny 立方体网格在 x-y 平面：
        cube (i, j) 中心 = ((i + 0.5) * cs, (j + 0.5) * cs, 0)
    射线沿 +x 方向穿过网格，会在 x = cs, 2*cs, ... 处遇到共面边界。
    """
    meshes = []
    for i in range(nx):
        for j in range(ny):
            c = trimesh.creation.box(extents=[cube_size, cube_size, cube_size])
            c.apply_translation([(i + 0.5) * cube_size,
                                 (j + 0.5) * cube_size, 0.0])
            meshes.append(c)
    combined = trimesh.util.concatenate(meshes)

    verts = torch.tensor(combined.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def make_heterogeneous_coplanar_x(device):
    """
    左侧: n=1.5 玻璃 (x ∈ [-1, 0])
    右侧: n=1.7 玻璃 (x ∈ [ 0, 1])
    共享 x=0 平面。

    ★ 关键：正确设置 x=0 处两个面的 face_materials。
        cube A 的 +x 面（法线朝 +x）外部是 cube B 的内部 (mat 2)
        cube B 的 -x 面（法线朝 -x）外部是 cube A 的内部 (mat 1)

    这样无论 tracer 命中哪个面，都能正确识别 n=1.5 → n=1.7。
    """
    import numpy as np

    a = trimesh.creation.box(extents=[1.0, 2.0, 2.0])
    a.apply_translation([-0.5, 0.0, 0.0])
    n_faces_a = len(a.faces)

    b = trimesh.creation.box(extents=[1.0, 2.0, 2.0])
    b.apply_translation([0.5, 0.0, 0.0])

    combined = trimesh.util.concatenate([a, b])
    verts_np = combined.vertices.copy()
    faces_np = combined.faces.copy()

    verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_np, dtype=torch.long, device=device)

    fm_np = np.zeros((len(faces_np), 2), dtype=np.int64)
    fm_np[:n_faces_a, 1] = 1   # cube A 内部 = mat 1 (n=1.5)
    fm_np[n_faces_a:, 1] = 2   # cube B 内部 = mat 2 (n=1.7)

    # ★ 手动修正 x=0 处的面
    for i in range(len(faces_np)):
        f = faces_np[i]
        if np.all(np.abs(verts_np[f, 0]) < 1e-6):
            # 这个面在 x=0 处
            if i < n_faces_a:
                # cube A 的 +x 面，外部是 cube B 的内部
                fm_np[i, 0] = 2
            else:
                # cube B 的 -x 面，外部是 cube A 的内部
                fm_np[i, 0] = 1

    fm = torch.tensor(fm_np, dtype=torch.long, device=device)
    return verts, faces, fm, n_faces_a


# ==============================================================================
# 光线构造
# ==============================================================================

def rays_along_x(device, n=64, x_start=-3.0, y_bounds=(-0.3, 0.3),
                  z_bounds=(-0.3, 0.3)):
    return generate_rays(
        source_type="grid", num_rays=n,
        origin=(x_start, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
        bounds=(y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]),
        polarization="s", device=device)


def rays_along_z(device, n=64, z_start=-3.0, x_bounds=(-0.3, 0.3),
                  y_bounds=(-0.3, 0.3)):
    return generate_rays(
        source_type="grid", num_rays=n,
        origin=(0.0, 0.0, z_start), direction=(0.0, 0.0, 1.0),
        bounds=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]),
        polarization="s", device=device)


def rays_through_grid_xy(device, n=64, x_start=-1.0, ny=3):
    """沿 +x 穿过 3×3 网格，光线起点 y 覆盖所有 j 位置。"""
    return generate_rays(
        source_type="grid", num_rays=n,
        origin=(x_start, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
        bounds=(0.3, ny - 0.3, -0.3, 0.3),
        polarization="s", device=device)


# ==============================================================================
# 诊断辅助
# ==============================================================================

def termination_steps(out):
    """返回每条光线的终止步数列表。"""
    active = out["active"]
    N = active.shape[0]
    steps = []
    for n in range(N):
        for s in range(active.shape[1]):
            if not active[n, s]:
                steps.append(s)
                break
        else:
            steps.append(active.shape[1] - 1)  # 未终止
    return steps


def assert_no_nan(out, tag=""):
    for key in ["positions", "stokes", "energy", "reflectivity",
                "transmissivity", "absorptivity"]:
        if key in out and isinstance(out[key], torch.Tensor):
            assert not torch.isnan(out[key]).any(), \
                f"{tag} {key} contains NaN"


def assert_all_terminated(out, tag="", max_steps=None):
    final_active = out["active"][:, -1]
    n_stuck = final_active.sum().item()
    assert n_stuck == 0, \
        f"{tag}: {n_stuck} rays did not terminate"


# ==============================================================================
# Test 1: 3 个立方体排成一行
# ==============================================================================

def test_three_cubes_in_a_row_x():
    """3 个玻璃立方体沿 +x 排成一行，射线穿过 3 个共面边界。"""
    device = torch.device("cpu")
    verts, faces, fm = make_chain_of_cubes_x(device, n_cubes=3)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = rays_along_x(device, n=64)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

    assert_no_nan(out, tag="[3-cube chain]")
    assert_all_terminated(out, tag="[3-cube chain]")
    assert out["energy_conserved"], "[3-cube chain] energy_conserved flag False"

    steps = termination_steps(out)
    print(f"\n[test 1] 3-cube chain: term steps "
          f"min={min(steps)} max={max(steps)} mean={sum(steps)/len(steps):.2f}")

    # 理论最小步数 = 4（进 cube 0，过 cube0/1 边界，过 cube1/2 边界，出 cube 2）
    # 实际可能稍多（+反射 / 跳过）
    assert max(steps) <= 12, \
        f"Too many steps for 3-cube chain: max={max(steps)} (expected ≤ 12)"


# ==============================================================================
# Test 2: 5 层堆叠
# ==============================================================================

def test_five_layers_stacked_z():
    """5 层玻璃板沿 +z 堆叠，射线穿过 5 层共面边界。"""
    device = torch.device("cpu")
    verts, faces, fm = make_tower_of_slabs_z(device, n_layers=5)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = rays_along_z(device, n=64)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=30, surface_model="fresnel")

    assert_no_nan(out, tag="[5-layer tower]")
    assert_all_terminated(out, tag="[5-layer tower]")
    assert out["energy_conserved"], "[5-layer tower] energy_conserved flag False"

    steps = termination_steps(out)
    print(f"\n[test 2] 5-layer tower: term steps "
          f"min={min(steps)} max={max(steps)} mean={sum(steps)/len(steps):.2f}")

    assert max(steps) <= 18, \
        f"Too many steps for 5-layer tower: max={max(steps)} (expected ≤ 18)"


# ==============================================================================
# Test 3: 3×3 网格
# ==============================================================================

def test_three_by_three_grid_of_cubes():
    """3×3 立方体网格，射线沿网格方向穿过 3 个共面边界。"""
    device = torch.device("cpu")
    verts, faces, fm = make_grid_of_cubes_xy(device, nx=3, ny=3)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = rays_through_grid_xy(device, n=64, x_start=-1.0, ny=3)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

    assert_no_nan(out, tag="[3x3 grid]")
    assert_all_terminated(out, tag="[3x3 grid]")
    assert out["energy_conserved"], "[3x3 grid] energy_conserved flag False"

    steps = termination_steps(out)
    print(f"\n[test 3] 3x3 grid along +x: term steps "
          f"min={min(steps)} max={max(steps)} mean={sum(steps)/len(steps):.2f}")

    assert max(steps) <= 12, \
        f"Too many steps for 3x3 grid: max={max(steps)}"


# ==============================================================================
# Test 4: 异构介质共面
# ==============================================================================

def test_heterogeneous_coplanar_interface():
    """
    两个不同折射率的玻璃共面接触：
        左: n=1.5, 右: n=1.7
    物理上穿过 x=0 时有 n=1.5 → n=1.7 的折射。

    断言：
      - 无 NaN
      - 所有光线终止
      - 能量守恒
      - 在 x=0 处至少有一个 n1≈1.5 且 n2≈1.7 的界面事件
    """
    device = torch.device("cpu")
    verts, faces, fm, n_faces_a = make_heterogeneous_coplanar_x(device)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        2: {"n": 1.7, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = rays_along_x(device, n=64, x_start=-3.0)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=15, surface_model="fresnel")

    assert_no_nan(out, tag="[hetero coplanar]")
    assert_all_terminated(out, tag="[hetero coplanar]")
    assert out["energy_conserved"], "[hetero coplanar] energy_conserved False"

    # ---- 找 n1 ≈ 1.5 且 n2 ≈ 1.7 的界面事件 ----
    n1 = out["n1"]   # [N, S+1, 1]
    n2 = out["n2"]
    hetero_mask = (torch.abs(n1 - 1.5) < 0.01) & (torch.abs(n2 - 1.7) < 0.01)
    n_hetero_events = hetero_mask.sum().item()

    print(f"\n[test 4] hetero n1=1.5 → n2=1.7 events: {n_hetero_events}")
    assert n_hetero_events > 0, \
        "No n=1.5 → n=1.7 interface event found — coplanar interface was missed!"

    # ---- 位置必须在 x ≈ 0 ----
    if n_hetero_events > 0:
        idx = torch.where(hetero_mask)
        # idx = (ray, step, _)
        xs = out["positions"][idx[0], idx[1], 0]
        print(f"[test 4] hetero interface x-positions: "
              f"min={xs.min().item():.4f}, max={xs.max().item():.4f}")
        assert torch.all(torch.abs(xs) < 1e-3), \
            f"hetero interface not at x=0: {xs.tolist()[:5]}"


# ==============================================================================
# Test 5: 8 个立方体长链（stress）
# ==============================================================================

def test_eight_cubes_chain():
    """8 个立方体链条，检验规模扩展性。"""
    device = torch.device("cpu")
    verts, faces, fm = make_chain_of_cubes_x(device, n_cubes=8)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    rays = rays_along_x(device, n=64, x_start=-3.0)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=40, surface_model="fresnel")

    assert_no_nan(out, tag="[8-cube chain]")
    assert_all_terminated(out, tag="[8-cube chain]")
    assert out["energy_conserved"], "[8-cube chain] energy_conserved False"

    steps = termination_steps(out)
    print(f"\n[test 5] 8-cube chain: term steps "
          f"min={min(steps)} max={max(steps)} mean={sum(steps)/len(steps):.2f}")

    # 8 个立方体 + 8 个共面边界 → 理论最少 10 步
    assert max(steps) <= 30, \
        f"Too many steps for 8-cube chain: max={max(steps)}"


# ==============================================================================
# Test 6: 多界面严格能量守恒
# ==============================================================================

def test_multi_object_energy_conservation():
    """
    每个界面事件必须满足 R + T + A ≈ 1，
    即使场景中有多个共面边界。

    使用 absorptivity=0.1 让 R / T / A 都非零，
    这样归一化逻辑的任何错误都会显现。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_chain_of_cubes_x(device, n_cubes=4)
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.1},
    }

    rays = rays_along_x(device, n=64, x_start=-3.0)
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

    R = out["reflectivity"]
    T = out["transmissivity"]
    A = out["absorptivity"]
    total = R + T + A

    # 只检查有交互的界面（total > 0）
    interacted = total > 1e-6
    n_events = interacted.sum().item()

    if n_events == 0:
        pytest.fail("No interface events detected")

    deviations = (total - 1.0)[interacted].abs()
    max_dev = deviations.max().item()
    mean_dev = deviations.mean().item()

    print(f"\n[test 6] 4-cube chain with absorption:")
    print(f"  Total interface events: {n_events}")
    print(f"  max |R+T+A-1| = {max_dev:.3e}")
    print(f"  mean |R+T+A-1| = {mean_dev:.3e}")

    assert max_dev < 1e-4, \
        f"Energy conservation violated at coplanar interfaces: " \
        f"max |R+T+A-1| = {max_dev:.6e}"

    assert out["energy_conserved"], "[multi-object] energy_conserved flag False"


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))