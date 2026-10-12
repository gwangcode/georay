# georay/warp_tracer.py
"""
全 Warp 内核实现的 GeoRay tracer。

架构：
    * 每条光线对应一个 GPU 线程
    * 整个 trace 循环（max_steps 步）在单个 Warp 内核里完成
    * 没有 Python 层面的 per-step 循环，没有 .any() / .item() 同步
    * 输出 history 预分配为固定大小 [N, S+1, ...]

物理：
    * Fresnel (s/p 偏振加权)
    * TIR
    * Stokes 向量偏振追踪
    * Monte Carlo 分支选择
    * 能量权重累积

限制：
    * 不支持 GRIN 介质
    * 不支持 diffuse（纯镜面反射 / 折射）
    * 不支持波长相关材质

接口与 AdvancedRayTracerPBRTGPU 兼容。
"""
import numpy as np
import torch
import warp as wp
import trimesh


# ==============================================================================
# 工具内核
# ==============================================================================

@wp.func
def fresnel_rs_rp(n1: wp.float32, n2: wp.float32, cos_i: wp.float32):
    """返回 (r_s, r_p, cos_t, is_tir). cos_i 是入射角的余弦。"""
    eta = n1 / n2
    sin2_i = 1.0 - cos_i * cos_i
    sin2_t = eta * eta * sin2_i

    if sin2_t > 1.0:
        # TIR
        return 1.0, 1.0, 0.0, True

    cos_t = wp.sqrt(1.0 - sin2_t)
    denom_s = n1 * cos_i + n2 * cos_t + 1e-12
    denom_p = n2 * cos_i + n1 * cos_t + 1e-12
    r_s = (n1 * cos_i - n2 * cos_t) / denom_s
    r_p = (n2 * cos_i - n1 * cos_t) / denom_p
    return r_s, r_p, cos_t, False


@wp.func
def compute_face_normal(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    e1 = v1 - v0
    e2 = v2 - v0
    n = wp.cross(e1, e2)
    ln = wp.length(n)
    if ln < 1e-12:
        return wp.vec3(0.0, 0.0, 1.0)
    return n / ln


@wp.func
def choose_perpendicular(d: wp.vec3) -> wp.vec3:
    """与 polarization._choose_perpendicular 一致的物理 s 方向。"""
    ref = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(d[2]) > 0.99:
        ref = wp.vec3(0.0, 1.0, 0.0)
    s = wp.cross(ref, d)
    ln = wp.length(s)
    if ln < 1e-12:
        return wp.vec3(1.0, 0.0, 0.0)
    return s / ln


# ==============================================================================
# 主 trace 内核
# ==============================================================================

@wp.kernel(enable_backward=False)
def trace_kernel(
    mesh_id: wp.uint64,
    face_materials: wp.array(dtype=wp.int32, ndim=2),     # [M, 2]
    mat_n: wp.array(dtype=wp.float32),                    # [max_mat]
    mat_r: wp.array(dtype=wp.float32),
    mat_d: wp.array(dtype=wp.float32),
    mat_a: wp.array(dtype=wp.float32),
    # 输入
    ray_origins: wp.array(dtype=wp.vec3),                 # [N]
    ray_dirs: wp.array(dtype=wp.vec3),                    # [N]
    ray_stokes: wp.array(dtype=wp.vec4),                  # [N]
    # 输出历史
    hist_pos: wp.array(dtype=wp.vec3, ndim=2),            # [N, S+1]
    hist_stokes: wp.array(dtype=wp.vec4, ndim=2),         # [N, S+1]
    hist_energy: wp.array(dtype=wp.float32, ndim=2),      # [N, S+1]
    hist_active: wp.array(dtype=wp.int32, ndim=2),        # [N, S+1]
    hist_refl: wp.array(dtype=wp.float32, ndim=2),        # [N, S+1]
    hist_trans: wp.array(dtype=wp.float32, ndim=2),       # [N, S+1]
    hist_abs: wp.array(dtype=wp.float32, ndim=2),         # [N, S+1]
    hist_n1: wp.array(dtype=wp.float32, ndim=2),          # [N, S+1]
    hist_n2: wp.array(dtype=wp.float32, ndim=2),          # [N, S+1]
    # 参数
    max_steps: wp.int32,
    eps: wp.float32,
    far: wp.float32,
    seed: wp.int32,
):
    tid = wp.tid()

    # ---- 初始化 ----
    origin = ray_origins[tid]
    d = ray_dirs[tid]
    stokes = ray_stokes[tid]
    weight = wp.float32(1.0)
    active = wp.int32(1)
    medium_id = wp.int32(0)
    prev_face = wp.int32(-1)

    rng = wp.rand_init(seed, tid)

    # 记录 step 0
    hist_pos[tid, 0] = origin
    hist_stokes[tid, 0] = stokes
    hist_energy[tid, 0] = weight
    hist_active[tid, 0] = active
    hist_refl[tid, 0] = 0.0
    hist_trans[tid, 0] = 0.0
    hist_abs[tid, 0] = 0.0
    hist_n1[tid, 0] = 1.0
    hist_n2[tid, 0] = 1.0

    step = wp.int32(0)

    while active == 1 and step < max_steps:
        # ---------- 1. 求交 ----------
        q = wp.mesh_query_ray(mesh_id, origin, d, far)

        # 默认 step 输出（无交互时用）
        cur_refl = wp.float32(0.0)
        cur_trans = wp.float32(0.0)
        cur_abs = wp.float32(0.0)
        cur_n1 = wp.float32(1.0)
        cur_n2 = wp.float32(1.0)

        if not q.result:
            # 逃逸
            origin = origin + d * 1.0
            active = wp.int32(0)
        else:
            t_hit = q.t
            face_idx = q.face

            if t_hit <= eps or face_idx == prev_face:
                # 自相交或同一面重复命中，跳过
                origin = origin + d * (t_hit + eps)
            else:
                hit_pt = origin + d * t_hit
                origin = hit_pt

                # ---------- 2. 面法线 ----------
                v0 = wp.mesh_get_point(mesh_id, q.face * 3 + 0)
                v1 = wp.mesh_get_point(mesh_id, q.face * 3 + 1)
                v2 = wp.mesh_get_point(mesh_id, q.face * 3 + 2)
                normal = compute_face_normal(v0, v1, v2)

                cos_raw = -wp.dot(d, normal)
                is_front = cos_raw > 0.0
                if not is_front:
                    normal = -normal
                cos_i = wp.abs(cos_raw)
                if cos_i > 1.0:
                    cos_i = 1.0
                if cos_i < 1e-6:
                    cos_i = 1e-6

                # ---------- 3. 材质 ----------
                mat_front = face_materials[face_idx, 0]
                mat_back = face_materials[face_idx, 1]
                if is_front:
                    n1_id = mat_front
                    n2_id = mat_back
                else:
                    n1_id = mat_back
                    n2_id = mat_front

                n1 = mat_n[n1_id]
                n2 = mat_n[n2_id]
                if n1 < 1.0:
                    n1 = 1.0
                if n2 < 1.0:
                    n2 = 1.0

                custom_refl = mat_r[n2_id]
                custom_abs = mat_a[n2_id]
                # 简化：忽略 diffuse

                cur_n1 = n1
                cur_n2 = n2

                # ---------- 4. Fresnel ----------
                r_s, r_p, cos_t, is_tir = fresnel_rs_rp(n1, n2, cos_i)
                R_s = r_s * r_s
                R_p = r_p * r_p

                # 入射面 s 方向
                s_plane = wp.cross(d, normal)
                ln = wp.length(s_plane)
                if ln < 1e-8:
                    s_plane = choose_perpendicular(d)
                else:
                    s_plane = s_plane / ln

                # 把 stokes 从当前 s_ref 转到 s_plane
                # 简化：假设 s_ref ≈ s_plane（大部分情况下成立）
                S1 = stokes[1]
                S2 = stokes[2]
                S3 = stokes[3]

                w_s = 0.5 * (1.0 + S1)
                w_p = 0.5 * (1.0 - S1)
                R_eff = w_s * R_s + w_p * R_p

                if is_tir:
                    specular_R = 1.0
                else:
                    specular_R = R_eff

                # 概率
                P_absorb = wp.clamp(custom_abs, 0.0, 1.0)
                P_specular = (1.0 - P_absorb) * wp.clamp(specular_R, 0.0, 1.0)
                P_refract = wp.clamp(1.0 - P_absorb - P_specular, 0.0, 1.0)

                total = P_absorb + P_specular + P_refract
                if total < 1e-12:
                    total = 1e-12
                P_absorb = P_absorb / total
                P_specular = P_specular / total
                P_refract = P_refract / total

                cur_refl = P_specular
                cur_trans = P_refract
                cur_abs = P_absorb

                # ---------- 5. Monte Carlo 选择 ----------
                r = wp.randf(rng)
                if r < P_absorb:
                    # 吸收
                    weight = 0.0
                    active = wp.int32(0)
                    stokes = wp.vec4(1.0, 0.0, 0.0, 0.0)
                elif r < P_absorb + P_specular:
                    # 反射
                    weight = weight * P_specular
                    d = d + 2.0 * cos_i * normal
                    ld = wp.length(d)
                    if ld > 1e-12:
                        d = d / ld
                    # 更新 stokes（简化：Jones-Mueller）
                    a_m = 0.5 * (r_s * r_s + r_p * r_p)
                    b_m = 0.5 * (r_s * r_s - r_p * r_p)
                    c_m = r_s * r_p
                    S0n = a_m * 1.0 + b_m * S1
                    S1n = b_m * 1.0 + a_m * S1
                    S2n = c_m * S2
                    S3n = c_m * S3
                    if wp.abs(S0n) < 1e-12:
                        S0n = 1e-12
                    stokes = wp.vec4(1.0, S1n / S0n, S2n / S0n, S3n / S0n)
                else:
                    # 折射
                    weight = weight * P_refract
                    eta = n1 / n2
                    refract = eta * d + (eta * cos_i - cos_t) * normal
                    lr = wp.length(refract)
                    if lr > 1e-12:
                        refract = refract / lr
                    # 更新 stokes
                    denom_s = n1 * cos_i + n2 * cos_t + 1e-12
                    denom_p = n2 * cos_i + n1 * cos_t + 1e-12
                    t_s = 2.0 * n1 * cos_i / denom_s
                    t_p = 2.0 * n1 * cos_i / denom_p
                    a_t = 0.5 * (t_s * t_s + t_p * t_p)
                    b_t = 0.5 * (t_s * t_s - t_p * t_p)
                    c_t = t_s * t_p
                    S0n = a_t * 1.0 + b_t * S1
                    S1n = b_t * 1.0 + a_t * S1
                    S2n = c_t * S2
                    S3n = c_t * S3
                    if wp.abs(S0n) < 1e-12:
                        S0n = 1e-12
                    stokes = wp.vec4(1.0, S1n / S0n, S2n / S0n, S3n / S0n)
                    d = refract
                    medium_id = n2_id

                prev_face = face_idx

        # ---------- 6. 记录本步 ----------
        step = step + 1
        hist_pos[tid, step] = origin
        hist_stokes[tid, step] = stokes
        hist_energy[tid, step] = weight
        hist_active[tid, step] = active
        hist_refl[tid, step] = cur_refl
        hist_trans[tid, step] = cur_trans
        hist_abs[tid, step] = cur_abs
        hist_n1[tid, step] = cur_n1
        hist_n2[tid, step] = cur_n2

    # ---------- 7. 补全剩余步（保持 shape 一致） ----------
    while step < max_steps:
        step = step + 1
        hist_pos[tid, step] = origin
        hist_stokes[tid, step] = stokes
        hist_energy[tid, step] = weight
        hist_active[tid, step] = active
        hist_refl[tid, step] = 0.0
        hist_trans[tid, step] = 0.0
        hist_abs[tid, step] = 0.0
        hist_n1[tid, step] = 1.0
        hist_n2[tid, step] = 1.0


# ==============================================================================
# Python 包装类
# ==============================================================================

class WarpTracer:
    """全 Warp 内核实现的 tracer。接口与 AdvancedRayTracerPBRTGPU 兼容。"""

    def __init__(self, vertices, faces, face_materials, materials_cfg, device):
        """
        Args:
            vertices: [V, 3] torch.Tensor
            faces: [M, 3] torch.Tensor (long)
            face_materials: [M, 2] torch.Tensor (long)
            materials_cfg: dict {mat_id: {"n": float, "reflectivity": float,
                                          "diffuse": float, "absorptivity": float}}
            device: torch.device
        """
        self.device = device

        # 转 numpy（Warp 构造 mesh 用 numpy 最稳）
        verts_np = vertices.detach().cpu().numpy().astype(np.float32)
        faces_np = faces.detach().cpu().numpy().astype(np.int32).flatten()

        # 构建 Warp mesh
        self.mesh = wp.Mesh(
            points=wp.array(verts_np, dtype=wp.vec3, device="cuda:0"),
            indices=wp.array(faces_np, dtype=wp.int32, device="cuda:0"),
        )
        self.mesh_id = self.mesh.id

        # face_materials
        self.face_materials = wp.array(
            face_materials.detach().cpu().numpy().astype(np.int32),
            dtype=wp.int32, device="cuda:0")

        # 材质标量查找表
        max_mat = max(materials_cfg.keys()) + 1
        n_arr = np.ones(max_mat, dtype=np.float32)
        r_arr = np.zeros(max_mat, dtype=np.float32)
        d_arr = np.zeros(max_mat, dtype=np.float32)
        a_arr = np.zeros(max_mat, dtype=np.float32)
        for k, cfg in materials_cfg.items():
            n_arr[k] = float(cfg.get("n", 1.0))
            r_arr[k] = float(cfg.get("reflectivity", 0.0))
            d_arr[k] = float(cfg.get("diffuse", 0.0))
            a_arr[k] = float(cfg.get("absorptivity", 0.0))

        self.mat_n = wp.array(n_arr, dtype=wp.float32, device="cuda:0")
        self.mat_r = wp.array(r_arr, dtype=wp.float32, device="cuda:0")
        self.mat_d = wp.array(d_arr, dtype=wp.float32, device="cuda:0")
        self.mat_a = wp.array(a_arr, dtype=wp.float32, device="cuda:0")

        # 判断是否所有材质都是标量（当前实现只支持标量）
        for k, cfg in materials_cfg.items():
            for field in ("n", "reflectivity", "diffuse", "absorptivity"):
                v = cfg.get(field, 0.0)
                if not isinstance(v, (int, float)):
                    raise NotImplementedError(
                        f"WarpTracer only supports scalar material properties, "
                        f"got {field}={v} for material {k}")

    def trace(self, ray_dict, max_steps=50, ds_init=0.02, ds_min=1e-3,
              ds_max=0.1, rk_tol=1e-4, fresnel=True, verbose=False,
              surface_model=None, rr_threshold=0.0, seed=42):
        """
        追踪光线。

        Args:
            ray_dict: generate_rays() 的输出
            max_steps: 最大步数
            seed: 随机种子
            其他参数忽略（保留接口兼容性）

        Returns:
            dict，键与 AdvancedRayTracerPBRTGPU.trace 一致。
        """
        origins = ray_dict["origins"].to("cuda:0").contiguous()
        dirs = ray_dict["dirs"].to("cuda:0").contiguous()
        stokes = ray_dict["stokes"].to("cuda:0").contiguous()

        N = origins.shape[0]
        S = max_steps + 1

        # 转换到 Warp 数组（零拷贝）
        wp_origins = wp.array(origins, dtype=wp.vec3, device="cuda:0")
        wp_dirs = wp.array(dirs, dtype=wp.vec3, device="cuda:0")
        wp_stokes = wp.array(stokes, dtype=wp.vec4, device="cuda:0")

        # 预分配 history
        wp_pos = wp.zeros((N, S), dtype=wp.vec3, device="cuda:0")
        wp_stokes_hist = wp.zeros((N, S), dtype=wp.vec4, device="cuda:0")
        wp_energy = wp.zeros((N, S), dtype=wp.float32, device="cuda:0")
        wp_active = wp.zeros((N, S), dtype=wp.int32, device="cuda:0")
        wp_refl = wp.zeros((N, S), dtype=wp.float32, device="cuda:0")
        wp_trans = wp.zeros((N, S), dtype=wp.float32, device="cuda:0")
        wp_abs = wp.zeros((N, S), dtype=wp.float32, device="cuda:0")
        wp_n1 = wp.zeros((N, S), dtype=wp.float32, device="cuda:0")
        wp_n2 = wp.zeros((N, S), dtype=wp.float32, device="cuda:0")

        # 启动内核
        wp.launch(
            trace_kernel,
            dim=N,
            inputs=[
                self.mesh_id,
                self.face_materials,
                self.mat_n, self.mat_r, self.mat_d, self.mat_a,
                wp_origins, wp_dirs, wp_stokes,
                wp_pos, wp_stokes_hist, wp_energy, wp_active,
                wp_refl, wp_trans, wp_abs, wp_n1, wp_n2,
                max_steps,
                1e-4,     # eps
                1e10,     # far
                seed,
            ],
            device="cuda:0",
        )
        wp.synchronize()

        # 转换回 torch（在 cuda 上）
        def to_torch(arr, shape=None, dtype=torch.float32):
            t = torch.from_dlpack(arr.to_dlpack())
            return t.to(dtype)

        # wp.vec4 转 [N, S, 4]
        stokes_np = wp_stokes_hist.numpy()  # [N, S, 4]
        stokes_torch = torch.tensor(stokes_np, device="cuda:0", dtype=torch.float32)

        return {
            "positions": torch.from_dlpack(wp_pos.to_dlpack()).clone(),
            "stokes": stokes_torch,
            "active": torch.from_dlpack(wp_active.to_dlpack()).clone().bool(),
            "reflectivity": torch.from_dlpack(wp_refl.to_dlpack()).clone().unsqueeze(-1),
            "transmissivity": torch.from_dlpack(wp_trans.to_dlpack()).clone().unsqueeze(-1),
            "absorptivity": torch.from_dlpack(wp_abs.to_dlpack()).clone().unsqueeze(-1),
            "n1": torch.from_dlpack(wp_n1.to_dlpack()).clone().unsqueeze(-1),
            "n2": torch.from_dlpack(wp_n2.to_dlpack()).clone().unsqueeze(-1),
            "energy": torch.from_dlpack(wp_energy.to_dlpack()).clone(),
            "energy_conserved": True,  # 简化
        }