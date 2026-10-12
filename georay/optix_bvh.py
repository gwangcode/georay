# georay/optix_bvh.py
"""
基于 trimesh-ray-optix (Triro) 的 OptiX 硬件光追加速器。

Triro 提供了与 trimesh.ray 兼容的接口，可以直接替换 GeoRay 中的
BVHMeshIntersector，同时利用 NVIDIA OptiX 的硬件 RT Core 实现
光线求交加速。

参考: https://github.com/lcp29/trimesh-ray-optix
"""
import torch
import trimesh
from triro.ray.ray_optix import RayMeshIntersector


class OptixMeshIntersector:
    """OptiX 加速的网格光线求交器，接口与 BVHMeshIntersector 兼容。"""

    def __init__(self, vertices, faces, device):
        """
        Args:
            vertices: torch.Tensor, shape [N, 3], float32, on device
            faces: torch.Tensor, shape [M, 3], int64, on device
            device: torch.device
        """
        self.device = device

        # Triro 的 RayMeshIntersector 支持直接传 vertices/faces
        # 内部会转为 float32 / int32 并放到 CUDA 上
        self.intersector = RayMeshIntersector(
            vertices=vertices.float().contiguous(),
            faces=faces.int().contiguous(),
        )

        # 预计算面法线 (与 BVHMeshIntersector 保持一致的物理计算)
        self.face_normals = self._compute_face_normals(vertices, faces)
        self.num_faces = faces.shape[0]

    @staticmethod
    def _compute_face_normals(vertices, faces):
        """计算面法线，与原 BVHMeshIntersector 保持一致的物理计算。"""
        tris = vertices[faces]
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        normals = torch.cross(e1, e2, dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-12)
        return normals.contiguous()

    def intersect(self, ray_origins, ray_dirs, prev_face_ids=None,
                  allowed_face_mask=None, eps=1e-4):
        """
        与 BVHMeshIntersector.intersect 完全兼容的接口。

        返回 (hit_dist, hit_face_ids, hit_normals)。
        """
        N = ray_origins.shape[0]

        ray_o = ray_origins.contiguous()
        ray_d = ray_dirs.contiguous()

        # ★ 使用 stream_compaction=False，保证输出与输入光线一一对应
        hit, front, tri_idx, location, uv = \
            self.intersector.intersects_closest(
                ray_o, ray_d, stream_compaction=False)

        # 初始化输出（与 BVHMeshIntersector 一致）
        hit_dist = torch.full((N,), float('inf'),
                              device=self.device, dtype=torch.float32)
        hit_face_ids = torch.full((N,), -1, dtype=torch.long,
                                  device=self.device)

        # 计算命中距离
        if hit.any():
            hit_locs = location[hit]            # [N_hit, 3]
            hit_origins = ray_origins[hit]      # [N_hit, 3]
            dists = torch.norm(hit_locs - hit_origins, dim=-1)

            hit_tri = tri_idx[hit].long()       # [N_hit,]

            # 距离过滤 + prev_face_ids 过滤
            valid_mask = (dists > eps)
            if prev_face_ids is not None:
                prev_hit = prev_face_ids[hit]
                valid_mask = valid_mask & (hit_tri != prev_hit)

            # allowed_face_mask 过滤
            if allowed_face_mask is not None:
                allowed = allowed_face_mask[hit_tri]
                valid_mask = valid_mask & allowed

            if valid_mask.any():
                valid_indices = torch.where(hit)[0][valid_mask]
                hit_dist[valid_indices] = dists[valid_mask]
                hit_face_ids[valid_indices] = hit_tri[valid_mask]

        # 获取命中面的法线
        safe_ids = torch.clamp(hit_face_ids, 0, self.num_faces - 1)
        hit_normals = self.face_normals[safe_ids]
        hit_normals[hit_face_ids < 0] = 0.0

        return hit_dist, hit_face_ids, hit_normals