"""
examples/demo_stepmesh2.py
==================================
使用 stepmesh.glue() 处理 cone.step / cube.step / cylinder.step，
用 GeoRay 追踪 10 条光线，导出 STL 供 FreeCAD 可视化。

前置安装：
    pip install git+https://github.com/gwangcode/stepmesh
    pip install git+https://github.com/gwangcode/georay

运行：
    cd georay/examples
    python demo_three_step_files.py

输出：
    ray_tracks.stl   -- 光线轨迹
    scene.stl        -- 场景几何
"""
import os
import numpy as np
import torch
import trimesh

try:
    from stepmesh import glue
except ImportError:
    raise ImportError(
        "请先安装 stepmesh:\n"
        "    pip install git+https://github.com/gwangcode/stepmesh")

from georay import AdvancedRayTracerPBRTGPU
from georay.stepmesh_adapter import glue_to_georay
from georay.io import export_ray_history_to_stl

import georay
print("[DEBUG] georay 包位置:", georay.__file__)


EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
STEP_FILES = [
    os.path.join(EXAMPLES_DIR, "cube.step"),
    os.path.join(EXAMPLES_DIR, "cone.step"),
    os.path.join(EXAMPLES_DIR, "cylinder.step"),
]

print("=" * 70)
print("GeoRay × stepmesh 演示：三个 STEP 文件的光线追踪")
print("=" * 70)

for f in STEP_FILES:
    assert os.path.exists(f), "找不到 STEP 文件: {}".format(f)
    print("  [OK] {}".format(os.path.basename(f)))


# ==============================================================================
# 1. 调用 stepmesh.glue() 处理 STEP 文件
# ==============================================================================

print("\n[1/6] 调用 stepmesh.glue() 处理 STEP 文件...")

try:
    result = glue(STEP_FILES)
except TypeError:
    result = glue(step_paths=STEP_FILES)

if isinstance(result, dict):
    vertices = np.asarray(result["vertices"])
    faces = np.asarray(result["faces"])
    mapping_table = np.asarray(result["mapping_table"])
    file_to_ids = result.get("file_to_ids_dict")
elif isinstance(result, (tuple, list)):
    vertices = np.asarray(result[0])
    faces = np.asarray(result[1])
    mapping_table = np.asarray(result[2])
    file_to_ids = result[3] if len(result) > 3 else None
else:
    vertices = np.asarray(result.vertices)
    faces = np.asarray(result.faces)
    mapping_table = np.asarray(result.mapping_table)
    file_to_ids = getattr(result, "file_to_ids_dict", None)

print("  vertices shape:      {}".format(vertices.shape))
print("  faces shape:         {}".format(faces.shape))
print("  mapping_table shape: {}".format(mapping_table.shape))


# ==============================================================================
# 2. 建立 体积ID → 材料ID 映射
# ==============================================================================

print("\n[2/6] 建立体积ID → 材料ID 映射...")

all_vols = sorted(set(mapping_table[:, 0].tolist()) |
                  set(mapping_table[:, 1].tolist()))
print("  所有 volume IDs: {}".format(all_vols))

vol_to_material = {0: 0}
object_materials = [1, 2, 3]
non_zero_vols = [v for v in all_vols if v != 0]
for i, v in enumerate(non_zero_vols):
    vol_to_material[v] = object_materials[i % len(object_materials)]

print("  vol_to_material = {}".format(vol_to_material))


# ==============================================================================
# 3. 转换为 GeoRay 张量
# ==============================================================================

print("\n[3/6] 转换为 GeoRay 张量...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("  device = {}".format(device))

verts_t, faces_t, face_mats_t = glue_to_georay(
    vertices=vertices,
    faces=faces,
    mapping_table=mapping_table,
    vol_to_material=vol_to_material,
    outside_material_id=0,
    device=device,
    remove_degenerate=True,
    verbose=True,
)


# ==============================================================================
# 4. 材料 + 生成 10 条光线
# ==============================================================================

print("\n[4/6] 配置材料 + 生成 10 条光线...")

materials_cfg = {
    0: {"n": 1.00, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.50, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    2: {"n": 1.77, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    3: {"n": 1.33, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
}

v_np = verts_t.detach().cpu().numpy()
bbox_min = v_np.min(axis=0)
bbox_max = v_np.max(axis=0)
bbox_center = 0.5 * (bbox_min + bbox_max)
bbox_size = bbox_max - bbox_min

print("  场景 bbox: min = {}".format(np.round(bbox_min, 3).tolist()))
print("             max = {}".format(np.round(bbox_max, 3).tolist()))
print("  场景中心:   {}".format(np.round(bbox_center, 3).tolist()))
print("  场景尺寸:   {}".format(np.round(bbox_size, 3).tolist()))

N_RAYS = 10
nx, ny = 2, 5
span_x = 0.3 * bbox_size[0]
span_y = 0.3 * bbox_size[1]

x_lin = np.linspace(bbox_center[0] - span_x, bbox_center[0] + span_x, nx)
y_lin = np.linspace(bbox_center[1] - span_y, bbox_center[1] + span_y, ny)
z_start = bbox_max[2] + 0.5 * bbox_size[2]

origins_np = np.zeros((N_RAYS, 3), dtype=np.float32)
dirs_np = np.zeros((N_RAYS, 3), dtype=np.float32)
k = 0
for x in x_lin:
    for y in y_lin:
        origins_np[k] = [x, y, z_start]
        dirs_np[k] = [0.0, 0.0, -1.0]
        k += 1

main_dir = torch.tensor([0.0, 0.0, -1.0], device=device)
ref_vec = torch.tensor([0.0, 1.0, 0.0], device=device)
u_vec = torch.cross(ref_vec, main_dir, dim=-1)
u_vec = u_vec / u_vec.norm()
v_vec = torch.cross(main_dir, u_vec, dim=-1)

rays = {
    "origins": torch.tensor(origins_np, dtype=torch.float32, device=device),
    "dirs": torch.tensor(dirs_np, dtype=torch.float32, device=device),
    "wavelengths": torch.full((N_RAYS, 1), 550.0,
                                dtype=torch.float32, device=device),
    "stokes": torch.tensor([[1.0, 1.0, 0.0, 0.0]] * N_RAYS,
                            dtype=torch.float32, device=device),
    "s_ref": u_vec.unsqueeze(0).expand(N_RAYS, 3).contiguous(),
    "p_ref": v_vec.unsqueeze(0).expand(N_RAYS, 3).contiguous(),
    "ray_ids": torch.arange(N_RAYS, dtype=torch.long, device=device),
    "medium_ids": torch.zeros(N_RAYS, dtype=torch.long, device=device),
}

print("  生成 {} 条光线".format(N_RAYS))
print("  x 范围: [{:.3f}, {:.3f}]".format(x_lin[0], x_lin[-1]))
print("  y 范围: [{:.3f}, {:.3f}]".format(y_lin[0], y_lin[-1]))
print("  z 起点: {:.3f}".format(z_start))


# ==============================================================================
# 5. 追踪
# ==============================================================================

print("\n[5/6] 执行光线追踪...")

tracer = AdvancedRayTracerPBRTGPU(
    verts_t, faces_t, face_mats_t, materials_cfg, device)

out = tracer.trace(
    rays,
    max_steps=30,
    ds_init=0.05,
    ds_min=1e-3,
    ds_max=0.2,
    surface_model="fresnel",
    fresnel=True,
)

print("  positions shape:  {}".format(tuple(out["positions"].shape)))
print("  active  shape:    {}".format(tuple(out["active"].shape)))
print("  energy_conserved: {}".format(out["energy_conserved"]))

# 打印每条光线的命中情况
print("\n  每条光线的命中情况：")
for i in range(N_RAYS):
    valid_len = int(out["active"][i].sum().item()) + 1
    final_pos = out["positions"][i, valid_len - 1].cpu().numpy()
    print("    Ray {:2d}: {} 步, 终止于 {}".format(
        i, valid_len - 1, np.round(final_pos, 3).tolist()))


# ★ 诊断：打印每条光线的每段长度
print("\n  每条光线的每段长度（诊断）：")
positions = out["positions"].cpu().numpy()
active = out["active"].cpu().numpy()
for i in range(N_RAYS):
    valid_len = int(active[i].sum()) + 1
    seg_lengths = []
    for j in range(valid_len - 1):
        seg = positions[i, j + 1] - positions[i, j]
        L = float(np.linalg.norm(seg))
        seg_lengths.append(L)
    print("    Ray {:2d}: {} 段, 各段长度 = {}".format(
        i, len(seg_lengths),
        [round(L, 2) for L in seg_lengths]))


# ==============================================================================
# 6. 导出 STL
# ==============================================================================

print("\n[6/6] 导出光线轨迹 STL...")

output_stl = os.path.join(EXAMPLES_DIR, "ray_tracks.stl")

# 半径 = bbox_max 的 0.002（原 0.008 的 1/4）
ray_radius = 0.002 * float(bbox_size.max())
print("  光线半径: {:.4f}".format(ray_radius))

export_ray_history_to_stl(
    out,
    filename=output_stl,
    radius=ray_radius,
    sides=6,
)
print("  光线 STL: {}".format(output_stl))

# 场景 STL
try:
    scene_stl = os.path.join(EXAMPLES_DIR, "scene.stl")
    scene_mesh = trimesh.Trimesh(
        vertices=verts_t.detach().cpu().numpy(),
        faces=faces_t.detach().cpu().numpy(),
    )
    scene_mesh.export(scene_stl)
    print("  场景 STL: {}".format(scene_stl))
except Exception as e:
    print("  场景 STL 导出失败: {}".format(e))


# ==============================================================================
# 7. 汇总
# ==============================================================================

print("\n" + "=" * 70)
print("完成")
print("=" * 70)
print("  ray_tracks.stl  -- {} 条光线轨迹".format(N_RAYS))
print("  scene.stl       -- 场景几何")
print()
print("  在 FreeCAD 中：")
print("    1. 打开 Part 工作台")
print("    2. File → Open → ray_tracks.stl")
print("    3. File → Open → scene.stl")
print("    4. View → Draw Style → Shaded")
print("=" * 70)