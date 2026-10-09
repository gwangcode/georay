"""BVH acceleration structure and triangle mesh intersector."""
import torch


class BVH:
    """SoA flattened BVH with pure PyTorch vectorized ray-triangle intersection."""

    def __init__(self, triangles: torch.Tensor, max_leaf_size: int = 4):
        self.device = triangles.device
        self.triangles = triangles.to(dtype=torch.float32)
        self.num_faces = self.triangles.shape[0]
        self.max_leaf_size = max_leaf_size

        self.tri_min = torch.min(self.triangles, dim=1)[0]
        self.tri_max = torch.max(self.triangles, dim=1)[0]
        self.tri_centroids = (self.tri_min + self.tri_max) * 0.5

        self.max_depth = 0
        self._build_bvh()

    def _build_bvh(self):
        nodes = []
        tri_indices = torch.arange(self.num_faces, device=self.device)
        reordered_indices = []

        def build_node(indices, depth=0):
            self.max_depth = max(self.max_depth, depth)
            num_tris = indices.shape[0]
            node_min = torch.min(self.tri_min[indices], dim=0)[0]
            node_max = torch.max(self.tri_max[indices], dim=0)[0]
            node_idx = len(nodes)
            nodes.append(None)

            if num_tris <= self.max_leaf_size:
                tri_start = len(reordered_indices)
                reordered_indices.extend(indices.tolist())
                nodes[node_idx] = [*node_min.tolist(), *node_max.tolist(),
                                   float(tri_start), float(num_tris), 1.0]
            else:
                extent = node_max - node_min
                axis = torch.argmax(extent).item()
                centroids = self.tri_centroids[indices, axis]
                median_val = torch.median(centroids)
                left_mask = centroids <= median_val
                if left_mask.all() or (~left_mask).all():
                    split_idx = num_tris // 2
                    left_indices = indices[:split_idx]
                    right_indices = indices[split_idx:]
                else:
                    left_indices = indices[left_mask]
                    right_indices = indices[~left_mask]
                left_child = build_node(left_indices, depth + 1)
                right_child = build_node(right_indices, depth + 1)
                nodes[node_idx] = [*node_min.tolist(), *node_max.tolist(),
                                   float(left_child), float(right_child), 0.0]
            return node_idx

        build_node(tri_indices)

        nodes_tensor = torch.tensor(nodes, dtype=torch.float32, device=self.device)
        self.node_min = nodes_tensor[:, :3].contiguous()
        self.node_max = nodes_tensor[:, 3:6].contiguous()
        self.left_or_start = nodes_tensor[:, 6].long().contiguous()
        self.right_or_count = nodes_tensor[:, 7].long().contiguous()
        self.is_leaf = (nodes_tensor[:, 8] > 0.5).contiguous()
        self.reordered_tri_ids = torch.tensor(reordered_indices, dtype=torch.long,
                                              device=self.device)
        self.ordered_triangles = self.triangles[self.reordered_tri_ids].contiguous()
        self.stack_capacity = max(32, min(512, self.max_depth * 2 + 16))

    @torch.no_grad()
    def ray_cast(self, ray_origins, ray_dirs, allowed_face_mask=None, eps=1e-4):
        N = ray_origins.shape[0]
        safe_dirs = torch.where(torch.abs(ray_dirs) < 1e-8,
                                torch.where(ray_dirs >= 0, torch.full_like(ray_dirs, 1e-8),
                                            torch.full_like(ray_dirs, -1e-8)),
                                ray_dirs)
        inv_dirs = 1.0 / safe_dirs

        min_hit_t = torch.full((N,), float('inf'), device=self.device)
        hit_face_idx = torch.full((N,), -1, dtype=torch.long, device=self.device)
        hit_u = torch.zeros((N,), device=self.device)
        hit_v = torch.zeros((N,), device=self.device)

        stack = torch.zeros((N, self.stack_capacity), dtype=torch.long, device=self.device)
        stack_ptr = torch.zeros((N,), dtype=torch.long, device=self.device)
        stack[:, 0] = 0
        stack_ptr += 1

        active_mask = stack_ptr > 0
        ray_ids = torch.arange(N, device=self.device)

        while active_mask.any():
            stack_ptr = torch.where(active_mask, stack_ptr - 1, stack_ptr)
            curr_node_idx = stack[ray_ids, stack_ptr]

            n_min = self.node_min[curr_node_idx]
            n_max = self.node_max[curr_node_idx]
            l_start = self.left_or_start[curr_node_idx]
            r_count = self.right_or_count[curr_node_idx]
            leaf_flag = self.is_leaf[curr_node_idx]

            t1 = (n_min - ray_origins) * inv_dirs
            t2 = (n_max - ray_origins) * inv_dirs
            tmin = torch.min(t1, t2)
            tmax = torch.max(t1, t2)
            t_near = torch.max(tmin, dim=-1)[0]
            t_far = torch.min(tmax, dim=-1)[0]
            hit_aabb = active_mask & (t_far >= torch.maximum(t_near, torch.zeros_like(t_near))) & (t_near < min_hit_t)

            leaf_hits = hit_aabb & leaf_flag
            if leaf_hits.any():
                hit_ray_indices = torch.where(leaf_hits)[0]
                starts = l_start[hit_ray_indices]
                counts = r_count[hit_ray_indices]
                ray_repeated = torch.repeat_interleave(hit_ray_indices, counts)
                offsets_base = torch.arange(self.max_leaf_size, device=self.device).unsqueeze(0)
                mask_offsets = offsets_base < counts.unsqueeze(-1)
                tri_offsets = offsets_base.expand(counts.shape[0], -1)[mask_offsets]
                tri_starts_rep = torch.repeat_interleave(starts, counts)
                flat_tri_indices = tri_starts_rep + tri_offsets

                o_flat = ray_origins[ray_repeated]
                d_flat = ray_dirs[ray_repeated]
                tris_flat = self.ordered_triangles[flat_tri_indices]
                tri_orig_ids_flat = self.reordered_tri_ids[flat_tri_indices]

                v0, v1, v2 = tris_flat[:, 0], tris_flat[:, 1], tris_flat[:, 2]
                e1, e2 = v1 - v0, v2 - v0
                pvec = torch.cross(d_flat, e2, dim=-1)
                det = torch.sum(e1 * pvec, dim=-1)
                valid_det = torch.abs(det) > 1e-10
                inv_det = 1.0 / torch.where(valid_det, det, torch.ones_like(det))
                tvec = o_flat - v0
                u = torch.sum(tvec * pvec, dim=-1) * inv_det
                qvec = torch.cross(tvec, e1, dim=-1)
                v = torch.sum(d_flat * qvec, dim=-1) * inv_det
                t = torch.sum(e2 * qvec, dim=-1) * inv_det

                valid = (valid_det & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) &
                         (u + v <= 1.0) & (t > eps) & (t < min_hit_t[ray_repeated]))
                if allowed_face_mask is not None:
                    valid = valid & allowed_face_mask[tri_orig_ids_flat]

                if valid.any():
                    v_ray_ids = ray_repeated[valid]
                    v_t = t[valid]
                    v_tri_ids = tri_orig_ids_flat[valid]
                    v_u = u[valid]
                    v_v = v[valid]

                    sort_perm = torch.argsort(v_t)
                    s_ray_ids = v_ray_ids[sort_perm]
                    s_t = v_t[sort_perm]
                    s_tri_ids = v_tri_ids[sort_perm]
                    s_u = v_u[sort_perm]
                    s_v = v_v[sort_perm]

                    unique_rays, inverse = torch.unique(s_ray_ids, return_inverse=True)
                    idx_all = torch.arange(s_ray_ids.shape[0], device=self.device)
                    first_indices = torch.full((unique_rays.shape[0],), s_ray_ids.shape[0],
                                               dtype=torch.long, device=self.device)
                    first_indices.scatter_reduce_(0, inverse, idx_all, reduce="amin",
                                                  include_self=True)

                    better_mask = s_t[first_indices] < min_hit_t[unique_rays]
                    target_rays = unique_rays[better_mask]
                    target_indices = first_indices[better_mask]

                    min_hit_t[target_rays] = s_t[target_indices]
                    hit_face_idx[target_rays] = s_tri_ids[target_indices]
                    hit_u[target_rays] = s_u[target_indices]
                    hit_v[target_rays] = s_v[target_indices]

            inner_hits = hit_aabb & (~leaf_flag)
            if inner_hits.any():
                can_push = stack_ptr < (self.stack_capacity - 1)
                push_mask = inner_hits & can_push
                stack[push_mask, stack_ptr[push_mask]] = l_start[push_mask]
                stack_ptr = torch.where(push_mask, stack_ptr + 1, stack_ptr)
                stack[push_mask, stack_ptr[push_mask]] = r_count[push_mask]
                stack_ptr = torch.where(push_mask, stack_ptr + 1, stack_ptr)

            active_mask = stack_ptr > 0

        hit_mask = (min_hit_t < float('inf')) & (hit_face_idx != -1)
        hit_face_ids = torch.where(hit_mask, hit_face_idx, torch.tensor(-1, device=self.device))
        hit_dist = torch.where(hit_mask, min_hit_t, torch.tensor(float('inf'), device=self.device))
        return hit_dist, hit_face_ids, hit_u, hit_v


class BVHMeshIntersector:
    """Mesh intersector with BVH acceleration."""

    def __init__(self, vertices, faces, device):
        self.device = device
        self.vertices = vertices.to(dtype=torch.float32, device=device)
        self.faces = faces.to(dtype=torch.long, device=device)
        self.triangles = self.vertices[self.faces]
        e1 = self.triangles[:, 1] - self.triangles[:, 0]
        e2 = self.triangles[:, 2] - self.triangles[:, 0]
        raw_normals = torch.cross(e1, e2, dim=-1)
        self.face_normals = raw_normals / (torch.norm(raw_normals, dim=-1, keepdim=True) + 1e-12)
        self.bvh = BVH(self.triangles)

    def intersect(self, ray_origins, ray_dirs, prev_face_ids=None,
                  allowed_face_mask=None, eps=1e-4):
        hit_dist, hit_face_ids, _, _ = self.bvh.ray_cast(
            ray_origins, ray_dirs, allowed_face_mask=allowed_face_mask)
        hit_mask = (hit_dist > eps) & (hit_dist < float('inf'))
        if prev_face_ids is not None:
            hit_mask = hit_mask & (hit_face_ids != prev_face_ids)
        hit_face_ids = torch.where(hit_mask, hit_face_ids, torch.tensor(-1, device=self.device))
        hit_dist = torch.where(hit_mask, hit_dist, torch.tensor(float('inf'), device=self.device))
        safe_ids = torch.clamp(hit_face_ids, 0, self.faces.shape[0] - 1)
        hit_normals = self.face_normals[safe_ids]
        return hit_dist, hit_face_ids, hit_normals