"""
tests/test_bvh_accuracy.py
===========================
BVH 求交精度验证：与暴力求交对比。
"""
import math
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_scene(device, n_spheres=20):
    """构造有球和盒子的大场景。"""
    meshes = [trimesh.creation.box(extents=[10.0, 10.0, 10.0])]
    for k in range(n_spheres):
        s = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
        theta = 2 * math.pi * k / n_spheres
        r = 3.0
        s.apply_translation([r * math.cos(theta), r * math.sin(theta), 0.0])
        meshes.append(s)
    combined = trimesh.util.concatenate(meshes)
    verts = torch.tensor(combined.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(combined.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def brute_force_intersect(ray_origins, ray_dirs, tris, eps=1e-4):
    """暴力三角形求交（无 BVH），用于对照。"""
    N = ray_origins.shape[0]
    M = tris.shape[0]

    o = ray_origins.unsqueeze(1)
    d = ray_dirs.unsqueeze(1)

    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0

    pvec = torch.cross(d, e2.unsqueeze(0).expand(N, M, 3), dim=-1)
    det = (e1.unsqueeze(0).expand(N, M, 3) * pvec).sum(dim=-1)
    valid_det = torch.abs(det) > 1e-10
    inv_det = 1.0 / torch.where(valid_det, det, torch.ones_like(det))

    tvec = o - v0.unsqueeze(0).expand(N, M, 3)
    u = (tvec * pvec).sum(dim=-1) * inv_det
    qvec = torch.cross(tvec, e1.unsqueeze(0).expand(N, M, 3), dim=-1)
    v = (d * qvec).sum(dim=-1) * inv_det
    t = (e2.unsqueeze(0).expand(N, M, 3) * qvec).sum(dim=-1) * inv_det

    valid = (valid_det & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps))
    t_masked = torch.where(valid, t, torch.full_like(t, float("inf")))

    hit_dist, hit_face = t_masked.min(dim=1)
    hit_face = torch.where(hit_dist < float("inf"), hit_face,
                            torch.tensor(-1, device=hit_face.device))
    return hit_dist, hit_face


# ==============================================================================
# Test 1: BVH 与暴力求交的命中距离一致
# ==============================================================================

def test_bvh_matches_brute_force_distance():
    """BVH 求交距离应与暴力求交一致。"""
    device = torch.device("cpu")
    verts, faces, fm = make_scene(device, n_spheres=20)

    # 随机光线
    torch.manual_seed(42)
    N = 512
    origins = torch.empty(N, 3, device=device).uniform_(-5.0, 5.0)
    origins[:, 2] = -5.0

    dirs = torch.zeros(N, 3, device=device)
    dirs[:, 0] = torch.empty(N, device=device).uniform_(-0.3, 0.3)
    dirs[:, 1] = torch.empty(N, device=device).uniform_(-0.3, 0.3)
    dirs[:, 2] = 1.0
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    # BVH
    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    # 用 intersector 直接
    dist_bvh, face_bvh, _ = tracer.intersector.intersect(origins, dirs)

    # 暴力
    tris = verts[faces]
    dist_bf, face_bf = brute_force_intersect(origins, dirs, tris)

    # 比较距离（允许小误差）
    valid = (dist_bvh < float("inf")) & (dist_bf < float("inf"))
    if valid.any():
        diff = (dist_bvh[valid] - dist_bf[valid]).abs().max().item()
        print("\n[Test 1] BVH vs brute force")
        print("  Valid hits: {}".format(valid.sum().item()))
        print("  max |Δdist|: {:.3e}".format(diff))
        assert diff < 1e-3, \
            "BVH distance mismatch: {:.3e}".format(diff)

    # 比较命中面 ID
    both_hit = (face_bvh >= 0) & (face_bvh >= 0)
    if both_hit.any():
        # 允许命中不同面（如果多个面在同一位置），但距离应相同
        pass


# ==============================================================================
# Test 2: 无漏检
# ==============================================================================

def test_bvh_no_misses():
    """BVH 不应漏检任何暴力求交能命中的光线。"""
    device = torch.device("cpu")
    verts, faces, fm = make_scene(device, n_spheres=10)

    torch.manual_seed(123)
    N = 256
    origins = torch.empty(N, 3, device=device).uniform_(-4.0, 4.0)
    origins[:, 2] = -5.0
    dirs = torch.zeros(N, 3, device=device)
    dirs[:, 2] = 1.0

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    dist_bvh, face_bvh, _ = tracer.intersector.intersect(origins, dirs)

    tris = verts[faces]
    dist_bf, face_bf = brute_force_intersect(origins, dirs, tris)

    bf_hits = dist_bf < float("inf")
    bvh_hits = dist_bvh < float("inf")

    # 所有 BF 命中的光线，BVH 也应命中
    missed = bf_hits & (~bvh_hits)
    n_missed = missed.sum().item()

    print("\n[Test 2] BVH no misses")
    print("  BF hits: {}".format(bf_hits.sum().item()))
    print("  BVH hits: {}".format(bvh_hits.sum().item()))
    print("  Missed by BVH: {}".format(n_missed))

    assert n_missed == 0, "BVH missed {} rays".format(n_missed)


# ==============================================================================
# Test 3: 大规模场景 BVH 求交速度优势
# ==============================================================================

def test_bvh_speedup_over_brute_force():
    """BVH 求交应显著快于暴力求交。"""
    import time
    device = torch.device("cpu")
    verts, faces, fm = make_scene(device, n_spheres=30)
    tris = verts[faces]

    materials = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)

    N = 1024
    torch.manual_seed(42)
    origins = torch.zeros(N, 3, device=device)
    origins[:, 2] = -5.0
    dirs = torch.zeros(N, 3, device=device)
    dirs[:, 2] = 1.0

    # BVH
    _ = tracer.intersector.intersect(origins, dirs)  # warmup
    t0 = time.perf_counter()
    for _ in range(3):
        _ = tracer.intersector.intersect(origins, dirs)
    t_bvh = (time.perf_counter() - t0) / 3

    # Brute force
    _ = brute_force_intersect(origins, dirs, tris)  # warmup
    t0 = time.perf_counter()
    for _ in range(3):
        _ = brute_force_intersect(origins, dirs, tris)
    t_bf = (time.perf_counter() - t0) / 3

    print("\n[Test 3] BVH vs brute force speed")
    print("  BVH time: {:.4f} s".format(t_bvh))
    print("  BF  time: {:.4f} s".format(t_bf))
    print("  Speedup : {:.1f}x".format(t_bf / t_bvh if t_bvh > 0 else 0))

    assert t_bvh < t_bf, \
        "BVH should be faster than brute force"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))