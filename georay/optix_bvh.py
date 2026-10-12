# georay/optix_bvh.py
import torch
import torchoptix

class OptixMeshIntersector:
    """使用 torchoptix 加速的网格光线求交器，接口与 BVHMeshIntersector 兼容。"""

    def __init__(self, vertices, faces, device):
        self.device = device
        self.num_faces = faces.shape[0]

        # torchoptix 需要 contiguous 的 GPU 张量
        self.vertices = vertices.to(dtype=torch.float32, device=device).contiguous()
        self.faces = faces.to(dtype=torch.int32, device=device).contiguous()

        # 构建 OptiX 加速结构
        self.handle = torchoptix.build(
            self.vertices.data_ptr(),
            self.faces.data_ptr(),
            len(self.vertices),
            len(self.faces)
        )

        # 预计算面法线 (保持和 BVHMeshIntersector 一致)
        tris = self.vertices[self.faces.long()]
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        normals = torch.cross(e1, e2, dim=-1)
        self.face_normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-12)

    def intersect(self, ray_origins, ray_dirs, prev_face_ids=None,
                  allowed_face_mask=None, eps=1e-4):
        N = ray_origins.shape[0]
        ray_o = ray_origins.contiguous()
        ray_d = ray_dirs.contiguous()

        out_t = torch.empty(N, device=self.device, dtype=torch.float32)
        out_i = torch.empty(N, device=self.device, dtype=torch.int32)

        far = 1e10
        torchoptix.trace_rays(
            self.handle,
            ray_o.data_ptr(),
            ray_d.data_ptr(),
            out_t.data_ptr(),
            out_i.data_ptr(),
            far,
            N
        )

        # torchoptix 命中时 out_i > 0 且 out_t < far
        hit_mask = (out_t < far) & (out_i > 0)

        hit_dist = torch.where(hit_mask, out_t,
                               torch.tensor(float('inf'), device=self.device))
        hit_face_ids = torch.where(hit_mask, out_i.long(),
                                   torch.tensor(-1, device=self.device))

        # 过滤：距离过小 或 来自 prev_face_ids
        valid = (hit_dist > eps)
        if prev_face_ids is not None:
            valid = valid & (hit_face_ids != prev_face_ids)
        if allowed_face_mask is not None:
            safe_ids = torch.clamp(hit_face_ids, 0, self.num_faces - 1)
            valid = valid & allowed_face_mask[safe_ids]

        hit_dist = torch.where(valid, hit_dist,
                               torch.tensor(float('inf'), device=self.device))
        hit_face_ids = torch.where(valid, hit_face_ids,
                                   torch.tensor(-1, device=self.device))

        safe_ids = torch.clamp(hit_face_ids, 0, self.num_faces - 1)
        hit_normals = self.face_normals[safe_ids]
        hit_normals[hit_face_ids < 0] = 0.0

        return hit_dist, hit_face_ids, hit_normals

    def __del__(self):
        if hasattr(self, 'handle') and self.handle is not None:
            torchoptix.release(self.handle)