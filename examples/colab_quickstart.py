"""
GeoRay Colab 快速入门
=====================
复制粘贴到 Google Colab 单元格运行。

前置：开启 GPU 运行时
    Colab 菜单 → 代码执行 → 更改运行时类型 → T4 GPU
"""

# ==============================================================================
# Cell 1: 安装
# ==============================================================================
# !pip install -q "georay[warp] @ git+https://github.com/gwangcode/georay"
# !pip install -q trimesh

# ==============================================================================
# Cell 2: 环境检查
# ==============================================================================
import torch
import warp as wp

assert torch.cuda.is_available(), "请开启 GPU 运行时！"
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import georay
print(f"GeoRay version: {georay.__version__}")
print(f"WarpTracer:     {'available' if georay.WarpTracer else 'NOT available'}")

# ==============================================================================
# Cell 3: 快速追踪示例
# ==============================================================================
import trimesh
from georay import WarpTracer, generate_rays

device = torch.device("cuda")

# 场景：玻璃立方体
cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
fm[:, 0] = 0
fm[:, 1] = 1
materials = {
    0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
}

# 光线：从空气沿 +z 入射
rays = generate_rays(
    source_type="grid", num_rays=1024,
    origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.5, 0.5, -0.5, 0.5),
    polarization="s", device=device)

# 追踪
tracer = WarpTracer(verts, faces, fm, materials, device)
out = tracer.trace(rays, max_steps=20, surface_model="fresnel")

print(f"positions:        {tuple(out['positions'].shape)}")
print(f"energy_conserved: {out['energy_conserved']}")

# ==============================================================================
# Cell 4: 从介质内部发射光线（新功能）
# ==============================================================================
from georay import generate_rays

# 从玻璃内部 z=0 出发
rays_inside = generate_rays(
    source_type="grid", num_rays=256,
    origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
    bounds=(-0.3, 0.3, -0.3, 0.3),
    polarization="s",
    medium_id=1,          # ★ 从玻璃（material 1）内部出发
    device=device)

out_inside = tracer.trace(rays_inside, max_steps=10)
print(f"从介质内部出发的追踪结果：")
print(f"  energy_conserved: {out_inside['energy_conserved']}")
print(f"  n1[step 1] = {out_inside['n1'][0, 1, 0].item():.3f}  (期望 1.5)")

# ==============================================================================
# Cell 5: 异常处理示例
# ==============================================================================
from georay import InvalidMaterialError, InvalidRayError

# 非法材质 → 立即捕获
try:
    bad_materials = {0: {"n": -1.5}}   # n < 0
    WarpTracer(verts, faces, fm, bad_materials, device)
except InvalidMaterialError as e:
    print(f"✓ 捕获到 InvalidMaterialError: {e}")

# 非法光线 → 立即捕获
try:
    bad_rays = {
        "origins": torch.zeros((4, 3), device=device),
        # 缺少 dirs 等字段
    }
    tracer.trace(bad_rays)
except InvalidRayError as e:
    print(f"✓ 捕获到 InvalidRayError: {e}")

# ==============================================================================
# Cell 6: 性能基准
# ==============================================================================
import time

for N in [256, 1024, 4096, 16384]:
    rays = generate_rays(
        source_type="grid", num_rays=N,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", device=device)

    tracer.trace(rays, max_steps=2)   # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tracer.trace(rays, max_steps=20)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"  N={N:6d}: {t_ms:8.2f} ms  ({N/t_ms*1000/1e6:.2f} M rays/s)")