"""
Demo: Trace rays through a multi-body STEP assembly remeshed with stepmesh.glue().

Prerequisites:
    pip install stepmesh
    三个 STEP 文件：cube.step, cone.step, cylinder.step（放在 examples/ 下）
"""
import os
import numpy as np
import torch

from georay import (
    AdvancedRayTracerPBRTGPU,
    generate_rays,
    save_trace_output_to_npz,
    export_ray_history_to_stl,
)
from georay.stepmesh_adapter import (
    glue_to_georay,
    summarize_glue_output,
    validate_georay_input,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # 1. 定位 STEP 文件
    # ---------------------------------------------------------------
    from stepmesh import glue

    examples_dir = os.path.dirname(os.path.abspath(__file__))
    step_files = [
        os.path.join(examples_dir, "cube.step"),
        os.path.join(examples_dir, "cone.step"),
        os.path.join(examples_dir, "cylinder.step"),
    ]

    for f in step_files:
        assert os.path.exists(f), "找不到 STEP 文件: {}".format(f)
        print(f"  [OK] {os.path.basename(f)}")

    # ---------------------------------------------------------------
    # 2. 调用 stepmesh.glue()
    # ---------------------------------------------------------------
    print("\n调用 stepmesh.glue()...")

    glue_result = glue(
        step_paths=step_files,
        min_size=0.5,
        max_size=2.0,
        return_internal_surfaces=True,
        return_each_tet_as_body=False,
    )

    if isinstance(glue_result, dict):
        vertices = np.asarray(glue_result["vertices"])
        faces = np.asarray(glue_result["faces"])
        mapping_table = np.asarray(glue_result["mapping_table"])
        file_to_ids_dict = glue_result.get("file_to_ids_dict", {})
    else:
        vertices = np.asarray(glue_result[0])
        faces = np.asarray(glue_result[1])
        mapping_table = np.asarray(glue_result[2])
        file_to_ids_dict = glue_result[3] if len(glue_result) > 3 else {}

    print(f"  vertices:      {vertices.shape}")
    print(f"  faces:         {faces.shape}")
    print(f"  mapping_table: {mapping_table.shape}")

    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("glue() 返回空几何，请检查 STEP 文件")

    summarize_glue_output(vertices, faces, mapping_table, file_to_ids_dict)

    # ---------------------------------------------------------------
    # 3. 建立体积ID → 材料ID 映射
    # ---------------------------------------------------------------
    all_vols = sorted(set(mapping_table[:, 0].tolist()) |
                      set(mapping_table[:, 1].tolist()))
    print(f"\n所有 volume IDs: {all_vols}")

    vol_to_material = {0: 0}
    object_materials = [1, 2, 3]
    non_zero_vols = [v for v in all_vols if v != 0]
    for i, v in enumerate(non_zero_vols):
        vol_to_material[v] = object_materials[i % len(object_materials)]

    print(f"vol_to_material = {vol_to_material}")

    # ---------------------------------------------------------------
    # 4. 转换为 GeoRay 输入
    # ---------------------------------------------------------------
    print("\n转换为 GeoRay 张量...")
    verts, faces_t, face_mats = glue_to_georay(
        vertices, faces, mapping_table,
        vol_to_material=vol_to_material,
        outside_material_id=0,
        device=device,
    )
    validate_georay_input(verts, faces_t, face_mats)

    # ---------------------------------------------------------------
    # 5. 材料配置
    # ---------------------------------------------------------------
    materials_cfg = {
        0: {"n": 1.00, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.50, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        2: {"n": 1.77, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        3: {"n": 1.33, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # ---------------------------------------------------------------
    # 6. 生成光线
    # ---------------------------------------------------------------
    v_np = verts.detach().cpu().numpy()
    bmin = v_np.min(axis=0)
    bmax = v_np.max(axis=0)
    center = 0.5 * (bmin + bmax)
    size = bmax - bmin

    print(f"\n场景 bbox: min={np.round(bmin, 3).tolist()}, "
          f"max={np.round(bmax, 3).tolist()}")
    print(f"场景尺寸: {np.round(size, 3).tolist()}")

    # ★ 光线数量：用完全平方数（grid 光源要求）
    N_RAYS = 25    # 5×5 网格

    # ★ 光线从 -z 方向射入
    z_start = float(bmin[2] - 0.5 * size[2])

    # ★★ 关键修复：bounds 是相对 origin 的偏移，不是绝对坐标！
    span_x = 0.4 * size[0]   # ±40% × size，覆盖物体
    span_y = 0.4 * size[1]

    rays = generate_rays(
        source_type="grid", num_rays=N_RAYS,
        origin=(float(center[0]), float(center[1]), z_start),
        direction=(0.0, 0.0, 1.0),
        # ★★ 修复：传偏移，不是绝对坐标
        bounds=(-span_x, span_x, -span_y, span_y),
        polarization="random",
        wavelength_nm=550.0,
        device=device,
    )

    rays["medium_ids"] = torch.zeros(rays["origins"].shape[0],
                                       dtype=torch.long, device=device)

    print(f"生成 {rays['origins'].shape[0]} 条光线")
    print(f"  z 起点:   {z_start:.3f}")
    print(f"  x 范围:   [{center[0] - span_x:.3f}, {center[0] + span_x:.3f}]")
    print(f"  y 范围:   [{center[1] - span_y:.3f}, {center[1] + span_y:.3f}]")

    # 检查光线是否覆盖物体
    print(f"  物体 x 范围: [{bmin[0]:.3f}, {bmax[0]:.3f}]")
    print(f"  物体 y 范围: [{bmin[1]:.3f}, {bmax[1]:.3f}]")

    # ---------------------------------------------------------------
    # 7. 追踪
    # ---------------------------------------------------------------
    print("\n执行追踪...")
    tracer = AdvancedRayTracerPBRTGPU(
        verts, faces_t, face_mats, materials_cfg, device)

    out = tracer.trace(
        rays, max_steps=30,
        ds_init=0.05, ds_min=1e-3, ds_max=0.2,
        rk_tol=1e-4, fresnel=True,
        surface_model="fresnel",
    )

    print(f"\n=== Trace result ===")
    print(f"  Energy conserved: {out['energy_conserved']}")
    print(f"  Positions shape:  {tuple(out['positions'].shape)}")

    # ★ 完整诊断
    print("\n  每条光线的命中情况：")
    n_hit = 0
    n_miss = 0
    pos = out["positions"].cpu().numpy()
    active = out["active"].cpu().numpy()
    for i in range(pos.shape[0]):
        valid_len = int(active[i].sum()) + 1
        if valid_len < 3:
            n_miss += 1
        else:
            n_hit += 1
    print(f"    命中物体: {n_hit} / {pos.shape[0]}")
    print(f"    未命中:   {n_miss} / {pos.shape[0]}")

    # 方向变化诊断（前 5 条）
    print("\n前 5 条光线的方向变化（验证折射）：")
    for i in range(min(5, pos.shape[0])):
        valid_len = int(active[i].sum()) + 1
        if valid_len < 3:
            print(f"  Ray {i}: 未命中（{valid_len} 步）")
            continue
        d_before = pos[i, 1] - pos[i, 0]
        d_before = d_before / (np.linalg.norm(d_before) + 1e-12)
        d_after = pos[i, 2] - pos[i, 1]
        d_after = d_after / (np.linalg.norm(d_after) + 1e-12)
        cos_angle = np.clip(np.dot(d_before, d_after), -1.0, 1.0)
        angle_change = np.degrees(np.arccos(cos_angle))
        print(f"  Ray {i}: 方向变化 {angle_change:.2f}°")

    # n1/n2 历史
    for i in range(min(3, pos.shape[0])):
        n1_hist = out["n1"][i, :, 0].cpu().tolist()
        n2_hist = out["n2"][i, :, 0].cpu().tolist()
        print(f"  Ray {i} n1 前 5 步: {[round(v, 3) for v in n1_hist[:5]]}")
        print(f"  Ray {i} n2 前 5 步: {[round(v, 3) for v in n2_hist[:5]]}")

    # ---------------------------------------------------------------
    # 8. 导出
    # ---------------------------------------------------------------
    output_npz = os.path.join(examples_dir, "stepmesh_trace.npz")
    output_stl = os.path.join(examples_dir, "stepmesh_trace.stl")

    save_trace_output_to_npz(out, output_npz)

    # ★ 修复：光线半径从 0.03 改回 0.005（细 6 倍）
    ray_radius = 0.005 * float(size.max())
    print(f"\n光线半径: {ray_radius:.4f} (0.005 × bbox_max)")
    export_ray_history_to_stl(out, output_stl,
                               radius=ray_radius, sides=8)

    # 场景 STL
    import trimesh
    scene_stl = os.path.join(examples_dir, "scene.stl")
    scene_mesh = trimesh.Trimesh(
        vertices=verts.detach().cpu().numpy(),
        faces=faces_t.detach().cpu().numpy(),
    )
    scene_mesh.export(scene_stl)

    print(f"\n✓ NPZ: {output_npz}")
    print(f"✓ STL: {output_stl}")
    print(f"✓ Scene: {scene_stl}")


if __name__ == "__main__":
    main()