"""
对照实验：用 trimesh 生成的 watertight 几何追踪。
如果这个正常，说明问题在 stepmesh。
"""
import os
import numpy as np
import torch
import trimesh

from georay import (
    AdvancedRayTracerPBRTGPU,
    generate_rays,
    export_ray_history_to_stl,
)


def main():
    device = torch.device("cpu")
    examples_dir = os.path.dirname(os.path.abspath(__file__))

    # ============================================================
    # 用 trimesh 直接生成三个 watertight 物体
    # ============================================================
    print("构建 watertight 场景...")

    # 立方体：在原点
    cube = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    cube.apply_translation([0.0, 0.0, 0.0])

    # 球体：在 x=+3 处
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.5)
    sphere.apply_translation([3.0, 0.0, 0.0])

    # 圆柱：在 x=-3 处
    cyl = trimesh.creation.cylinder(radius=1.0, height=3.0)
    cyl.apply_translation([-3.0, 0.0, 0.0])

    # 分别保存以便材质不同
    # 使用 concatenate 构造场景，但需要手动标记每个物体的面
    n_cube = len(cube.faces)
    n_sphere = len(sphere.faces)
    n_cyl = len(cyl.faces)

    combined = trimesh.util.concatenate([cube, sphere, cyl])
    print(f"  总顶点: {len(combined.vertices)}")
    print(f"  总面:   {len(combined.faces)}")
    print(f"  watertight: {combined.is_watertight}")

    # 构造 face_mats
    # cube → material 1 (n=1.5)
    # sphere → material 2 (n=1.77)
    # cyl → material 3 (n=1.33)
    faces_t = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((len(combined.faces), 2), dtype=torch.long, device=device)
    fm[0:n_cube, 0] = 0
    fm[0:n_cube, 1] = 1
    fm[n_cube:n_cube+n_sphere, 0] = 0
    fm[n_cube:n_cube+n_sphere, 1] = 2
    fm[n_cube+n_sphere:, 0] = 0
    fm[n_cube+n_sphere:, 1] = 3

    verts_t = torch.tensor(combined.vertices, dtype=torch.float32, device=device)

    # ============================================================
    # 材料
    # ============================================================
    materials_cfg = {
        0: {"n": 1.00, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.50, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        2: {"n": 1.77, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        3: {"n": 1.33, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # ============================================================
    # 光线
    # ============================================================
    v_np = combined.vertices
    bmin = v_np.min(axis=0)
    bmax = v_np.max(axis=0)
    center = 0.5 * (bmin + bmax)
    size = bmax - bmin
    print(f"\n场景 bbox: min={np.round(bmin, 3).tolist()}, "
          f"max={np.round(bmax, 3).tolist()}")

    # 光线从 -z 方向射入
    N_RAYS = 25  # 5×5
    z_start = float(bmin[2] - 0.5 * size[2])
    span_x = 0.5 * size[0]
    span_y = 0.5 * size[1]

    rays = generate_rays(
        source_type="grid", num_rays=N_RAYS,
        origin=(float(center[0]), float(center[1]), z_start),
        direction=(0.0, 0.0, 1.0),
        bounds=(-span_x, span_x, -span_y, span_y),
        polarization="s",
        wavelength_nm=550.0,
        device=device,
    )
    rays["medium_ids"] = torch.zeros(N_RAYS, dtype=torch.long, device=device)

    # ============================================================
    # 追踪
    # ============================================================
    print("\n执行追踪...")
    tracer = AdvancedRayTracerPBRTGPU(
        verts_t, faces_t, fm, materials_cfg, device)
    out = tracer.trace(
        rays, max_steps=30, ds_init=0.05,
        ds_min=1e-3, ds_max=0.2,
        rk_tol=1e-4, fresnel=True, surface_model="fresnel",
    )

    # ============================================================
    # 诊断
    # ============================================================
    print(f"\n=== 追踪结果 ===")
    print(f"  Energy conserved: {out['energy_conserved']}")
    print(f"  Positions shape:  {tuple(out['positions'].shape)}")

    pos = out["positions"].cpu().numpy()
    active = out["active"].cpu().numpy()

    n_hit, n_miss = 0, 0
    for i in range(pos.shape[0]):
        valid_len = int(active[i].sum()) + 1
        if valid_len < 3:
            n_miss += 1
        else:
            n_hit += 1
    print(f"\n  命中物体: {n_hit} / {pos.shape[0]}")
    print(f"  未命中:   {n_miss} / {pos.shape[0]}")

    # 方向变化
    print("\n  方向变化（前 10 条命中光线）：")
    shown = 0
    for i in range(pos.shape[0]):
        valid_len = int(active[i].sum()) + 1
        if valid_len < 3:
            continue
        d_before = pos[i, 1] - pos[i, 0]
        d_before = d_before / (np.linalg.norm(d_before) + 1e-12)
        d_after = pos[i, 2] - pos[i, 1]
        d_after = d_after / (np.linalg.norm(d_after) + 1e-12)
        cos_a = np.clip(np.dot(d_before, d_after), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        n1_hist = out["n1"][i, :, 0].cpu().tolist()
        n2_hist = out["n2"][i, :, 0].cpu().tolist()
        print(f"  Ray {i}: 方向变化 {angle:.2f}°, "
              f"n1[1:5]={[round(v, 2) for v in n1_hist[1:5]]}, "
              f"n2[1:5]={[round(v, 2) for v in n2_hist[1:5]]}")
        shown += 1
        if shown >= 10:
            break

    # ============================================================
    # 导出
    # ============================================================
    output_stl = os.path.join(examples_dir, "watertight_trace.stl")
    ray_radius = 0.005 * float(size.max())
    print(f"\n光线半径: {ray_radius:.4f}")
    export_ray_history_to_stl(out, output_stl,
                               radius=ray_radius, sides=8)

    scene_stl = os.path.join(examples_dir, "watertight_scene.stl")
    combined.export(scene_stl)

    print(f"\n✓ STL: {output_stl}")
    print(f"✓ Scene: {scene_stl}")


if __name__ == "__main__":
    main()