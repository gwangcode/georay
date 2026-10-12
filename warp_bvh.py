# georay/warp_bvh.py
"""Warp 实现的 BVH 求交器，接口与 BVHMeshIntersector 兼容。"""
import numpy as np
import torch
import warp as wp


@wp.kernel(enable_backward=False)
def _warp_intersect_kernel(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    far: wp.float32,
    hit_dist: wp.array(dtype=wp.float32),
    hit_face: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh_id, origins[i], dirs[i], far)
    if q.result:
        hit_dist[i] = q.t
        hit_face[i] = q.face
    else:
        hit_dist[i] = far
        hit_face[i] = -1


class WarpMeshIntersector:
    """Warp 加速的 BVH 求交器。"""

    def __init__(self, vertices, faces, device):
        self.device = device
        self.warp_device = "cuda:0" if device.type == "cuda" else "cpu"

        if not wp.is_initialized():
            wp.init()

        verts_np = vertices.detach().cpu().numpy().astype(np.float32)
        faces_np = faces.detach().cpu().numpy().astype(np.int32).flatten()

        self.mesh = wp.Mesh(
            points=wp.array(verts_np, dtype=wp.vec3, device=self.warp_device),
            indices=wp.array(faces_np, dtype=wp.int32, device=self.warp_device),
        )
        self.mesh_id = self.mesh.id

        # 预计算面法线
        tris = vertices[faces]
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        n = torch.cross(e1, e2, dim=-1)
        self.face_normals = (n / (n.norm(dim=-1, keepdim=True) + 1e-12)).contiguous()
        self.num_faces = faces.shape[0]

    def intersect(self, ray_origins, ray_dirs, prev_face_ids=None,
                  allowed_face_mask=None, eps=1e-4):
        N = ray_origins.shape[0]
        o = ray_origins.detach().contiguous().float()
        d = ray_dirs.detach().contiguous().float()

        wp_o = wp.from_torch(o, dtype=wp.vec3)
        wp_d = wp.from_torch(d, dtype=wp.vec3)

        wp_t = wp.zeros(N, dtype=wp.float32, device=self.warp_device)
        wp_f = wp.zeros(N, dtype=wp.int32, device=self.warp_device)

        wp.launch(
            _warp_intersect_kernel,
            dim=N,
            inputs=[self.mesh_id, wp_o, wp_d, np.float32(1e10), wp_t, wp_f],
            device=self.warp_device,
        )
        wp.synchronize()

        hit_dist = wp.to_torch(wp_t).clone()
        hit_face_ids = wp.to_torch(wp_f).clone().long()

        hit_mask = (hit_dist > eps) & (hit_dist < 1e9)
        if prev_face_ids is not None:
            hit_mask = hit_mask & (hit_face_ids != prev_face_ids)
        if allowed_face_mask is not None:
            safe_ids = torch.clamp(hit_face_ids, 0, self.num_faces - 1)
            hit_mask = hit_mask & allowed_face_mask[safe_ids]

        hit_dist = torch.where(hit_mask, hit_dist,
                               torch.tensor(float('inf'), device=self.device))
        hit_face_ids = torch.where(hit_mask, hit_face_ids,
                                   torch.tensor(-1, device=self.device, dtype=torch.long))

        safe_ids = torch.clamp(hit_face_ids, 0, self.num_faces - 1)
        hit_normals = self.face_normals[safe_ids]
        hit_normals[hit_face_ids < 0] = 0.0

        return hit_dist, hit_face_ids, hit_normals