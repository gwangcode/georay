"""Main ray tracer class for GeoRay."""
import math
import torch
import torch.nn as nn

from .bvh import BVHMeshIntersector
from .interpolation import create_pure_torch_interpolated_cfg
from .grin import parse_grin_input_to_field
from .integrator import adaptive_rk4_step_per_ray
from .polarization import (
    _choose_perpendicular,
    rotate_stokes_reference,
    apply_jones_mueller,
    randomize_stokes_diffuse,
    stokes_to_pol_vector,
    degree_of_polarization,
)


class AdvancedRayTracerPBRTGPU:
    """GPU-accelerated ray tracer with BVH, GRIN, and polarization support.

    Supports:
        - BVH-accelerated triangle mesh intersection
        - Adaptive RK4 for gradient-index media
        - Full Stokes polarization tracking via Jones-Mueller formalism
        - Fresnel reflection/refraction with polarization-weighted reflectance
        - TIR, diffuse, absorption
        - Energy conservation checks
        - Rays starting inside a dielectric or GRIN medium (via ray_dict['medium_ids'])
    """

    def __init__(self, vertices, faces, face_materials, materials_cfg, device):
        self.device = device
        self.intersector = BVHMeshIntersector(vertices, faces, device)
        self.face_materials = face_materials.to(dtype=torch.long, device=device)
        self.materials_cfg = materials_cfg
        self.num_faces = faces.shape[0]

        # Build medium -> face mask lookup
        self.medium_to_faces_mask = {}
        all_medium_ids = set()
        for f_idx in range(self.num_faces):
            all_medium_ids.add(self.face_materials[f_idx, 0].item())
            all_medium_ids.add(self.face_materials[f_idx, 1].item())
        for med_id in all_medium_ids:
            mask = ((self.face_materials[:, 0] == med_id) |
                    (self.face_materials[:, 1] == med_id)).to(device)
            self.medium_to_faces_mask[med_id] = mask

        # Precompile material evaluators and GRIN fields
        self.mat_evaluators = {}
        self.grin_fields = {}
        for mat_id, cfg in materials_cfg.items():
            self.mat_evaluators[mat_id] = {
                "n": create_pure_torch_interpolated_cfg(cfg.get("n", 1.0), device),
                "reflectivity": create_pure_torch_interpolated_cfg(
                    cfg.get("reflectivity", 0.0), device),
                "diffuse": create_pure_torch_interpolated_cfg(
                    cfg.get("diffuse", 0.0), device),
                "absorptivity": create_pure_torch_interpolated_cfg(
                    cfg.get("absorptivity", 0.0), device),
            }
            if "grin_field" in cfg and cfg["grin_field"] is not None:
                self.grin_fields[mat_id] = parse_grin_input_to_field(
                    cfg["grin_field"], device)

        max_med = max(all_medium_ids) if all_medium_ids else 0
        self.grin_medium_mask = torch.zeros(max_med + 1, dtype=torch.bool,
                                            device=device)
        for med_id in self.grin_fields.keys():
            self.grin_medium_mask[med_id] = True

        self.max_medium_id = max_med

        # Precompute scalar material lookup table
        self.max_mat_id = max(materials_cfg.keys()) if materials_cfg else 0
        self.all_simple = self._check_all_simple(materials_cfg)
        if self.all_simple:
            M = self.max_mat_id + 1
            self.n_table = torch.ones(M, device=device)
            self.r_table = torch.zeros(M, device=device)
            self.d_table = torch.zeros(M, device=device)
            self.a_table = torch.zeros(M, device=device)
            for k, cfg in materials_cfg.items():
                self.n_table[k] = float(cfg.get("n", 1.0))
                self.r_table[k] = float(cfg.get("reflectivity", 0.0))
                self.d_table[k] = float(cfg.get("diffuse", 0.0))
                self.a_table[k] = float(cfg.get("absorptivity", 0.0))

    @staticmethod
    def _check_all_simple(materials_cfg):
        for k, cfg in materials_cfg.items():
            for field in ("n", "reflectivity", "diffuse", "absorptivity"):
                v = cfg.get(field, 0.0)
                if not isinstance(v, (int, float)):
                    return False
        return True

    def trace(self, ray_dict, max_steps=50, ds_init=0.02, ds_min=1e-3,
              ds_max=0.1, rk_tol=1e-4, fresnel=True, verbose=False,
              surface_model=None, rr_threshold=0.0):
        """
        Trace rays through the scene.

        Args:
            ray_dict: output of generate_rays().
                Optional key "medium_ids": [N] long tensor, initial medium
                ID per ray. Defaults to all zeros (air/vacuum).
            max_steps, ds_init, ds_min, ds_max, rk_tol: integrator params.
            fresnel: legacy flag.
            surface_model: "fresnel" | "custom" | "coated".
            verbose: print GRIN step diagnostics.
            rr_threshold: Russian roulette threshold (0 disables).

        Returns:
            dict with keys: positions, stokes, active, wavelengths,
                            reflectivity, transmissivity, absorptivity,
                            n1, n2, polarization_deg, dop,
                            energy, rk_error, ds_history, energy_conserved.
        """
        if surface_model is None:
            surface_model = "coated" if fresnel else "custom"
        if surface_model not in ("fresnel", "custom", "coated"):
            raise ValueError(
                f"surface_model must be 'fresnel', 'custom', or 'coated', "
                f"got '{surface_model}'")

        # ---- Initialize state ----
        curr_origins = ray_dict["origins"].clone()
        curr_dirs = ray_dict["dirs"].clone()
        curr_wls = ray_dict["wavelengths"].clone()
        curr_stokes = ray_dict["stokes"].clone()
        curr_s_ref = ray_dict["s_ref"].clone()
        curr_p_ref = ray_dict["p_ref"].clone()

        N = curr_origins.shape[0]

        # ★ 支持 ray_dict["medium_ids"]：每条光线的初始介质
        if "medium_ids" in ray_dict:
            mid = ray_dict["medium_ids"]
            if isinstance(mid, int):
                curr_medium_ids = torch.full((N,), int(mid), dtype=torch.long,
                                              device=self.device)
            else:
                curr_medium_ids = mid.to(dtype=torch.long, device=self.device).clone()
                if curr_medium_ids.shape[0] != N:
                    raise ValueError(
                        f"ray_dict['medium_ids'] length {curr_medium_ids.shape[0]} "
                        f"does not match number of rays {N}")
        else:
            curr_medium_ids = torch.zeros(N, dtype=torch.long, device=self.device)

        prev_hit_faces = torch.full((N,), -1, dtype=torch.long, device=self.device)
        active_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        curr_weight = torch.ones(N, device=self.device)
        ds_per_ray = torch.full((N,), float(ds_init), device=self.device)

        # ---- History buffers ----
        hist_pos = [curr_origins.clone()]
        hist_stokes = [curr_stokes.clone()]
        hist_active = [active_mask.clone()]
        hist_refl = [torch.zeros((N, 1), device=self.device)]
        hist_trans = [torch.zeros((N, 1), device=self.device)]
        hist_abs = [torch.zeros((N, 1), device=self.device)]
        hist_n1 = [torch.ones((N, 1), device=self.device)]
        hist_n2 = [torch.ones((N, 1), device=self.device)]
        hist_pol_deg = [0.5 * torch.atan2(curr_stokes[:, 2:3],
                                          curr_stokes[:, 1:2]) * (180.0 / math.pi)]
        hist_energy = [curr_weight.clone()]
        hist_rk_err = [torch.zeros(N, device=self.device)]
        hist_ds = [ds_per_ray.clone()]
        hist_dop = [degree_of_polarization(curr_stokes).clone()]

        step = 0
        while step < max_steps:
            if not active_mask.any():
                break

            # ---------- 1. Intersection ----------
            hit_dist = torch.full((N,), float('inf'), device=self.device)
            hit_face_ids = torch.full((N,), -1, dtype=torch.long, device=self.device)
            hit_normals = torch.zeros((N, 3), device=self.device)

            for med_val in range(self.max_medium_id + 1):
                ray_mask = active_mask & (curr_medium_ids == med_val)
                if not ray_mask.any():
                    continue
                allowed_faces = self.medium_to_faces_mask.get(med_val, None)
                dist_m, face_ids_m, normals_m = self.intersector.intersect(
                    curr_origins[ray_mask], curr_dirs[ray_mask],
                    prev_face_ids=prev_hit_faces[ray_mask],
                    allowed_face_mask=allowed_faces)
                hit_dist[ray_mask] = dist_m
                hit_face_ids[ray_mask] = face_ids_m
                hit_normals[ray_mask] = normals_m

            # ---------- 2. Buffers ----------
            next_origins = curr_origins.clone()
            next_dirs = curr_dirs.clone()
            next_stokes = curr_stokes.clone()
            next_s_ref = curr_s_ref.clone()
            next_p_ref = curr_p_ref.clone()
            next_weight = curr_weight.clone()

            step_refl = torch.zeros((N, 1), device=self.device)
            step_trans = torch.zeros((N, 1), device=self.device)
            step_abs = torch.zeros((N, 1), device=self.device)
            step_n1 = torch.ones((N, 1), device=self.device)
            step_n2 = torch.ones((N, 1), device=self.device)
            step_rk_err = torch.zeros(N, device=self.device)

            # ---------- 3. GRIN stepping ----------
            is_in_grin = self.grin_medium_mask[curr_medium_ids]
            grin_mask = active_mask & is_in_grin & (hit_dist > ds_per_ray)

            if grin_mask.any():
                pol_vec_all = stokes_to_pol_vector(curr_stokes, curr_s_ref, curr_p_ref)
                dop_all = degree_of_polarization(curr_stokes)

                if len(self.grin_fields) == 1:
                    g_val, field = next(iter(self.grin_fields.items()))
                    sub_mask = grin_mask & (curr_medium_ids == g_val)
                    if sub_mask.any():
                        field.set_context(pol_vec_all[sub_mask], dop_all[sub_mask])
                        h_sub = ds_per_ray[sub_mask]
                        r_new, d_new, n_eval, h_new, err_new, accepted = \
                            adaptive_rk4_step_per_ray(
                                curr_origins[sub_mask], curr_dirs[sub_mask], field,
                                curr_wls[sub_mask], h_per_ray=h_sub,
                                tol=rk_tol, h_min=ds_min, h_max=ds_max)
                        s_old = curr_s_ref[sub_mask]
                        s_proj = torch.sum(s_old * d_new, dim=-1, keepdim=True)
                        s_candidate = s_old - s_proj * d_new
                        s_candidate = s_candidate / (
                                torch.norm(s_candidate, dim=-1, keepdim=True) + 1e-12)
                        stokes_candidate = rotate_stokes_reference(
                            curr_stokes[sub_mask], s_old, s_candidate, d_new)
                        acc_3d = accepted.unsqueeze(-1)
                        next_origins[sub_mask] = torch.where(
                            acc_3d, r_new, curr_origins[sub_mask])
                        next_dirs[sub_mask] = torch.where(
                            acc_3d, d_new, curr_dirs[sub_mask])
                        next_s_ref[sub_mask] = torch.where(
                            acc_3d, s_candidate, curr_s_ref[sub_mask])
                        next_p_ref[sub_mask] = torch.cross(
                            next_dirs[sub_mask], next_s_ref[sub_mask], dim=-1)
                        next_stokes[sub_mask] = torch.where(
                            acc_3d, stokes_candidate, curr_stokes[sub_mask])
                        n_eval_col = n_eval if n_eval.dim() == 2 else n_eval.unsqueeze(-1)
                        step_n1[sub_mask] = torch.where(
                            acc_3d, n_eval_col, step_n1[sub_mask])
                        step_n2[sub_mask] = torch.where(
                            acc_3d, n_eval_col, step_n2[sub_mask])
                        step_rk_err[sub_mask] = err_new
                        ds_per_ray[sub_mask] = h_new
                        field.clear_context()
                        if verbose and step % 5 == 0:
                            print(f"  [GRIN step={step} med={g_val}] "
                                  f"ds=[{h_new.min().item():.5f}, "
                                  f"{h_new.max().item():.5f}], "
                                  f"acc={accepted.float().mean().item():.1%}")
                else:
                    grin_meds = torch.unique(curr_medium_ids[grin_mask])
                    for g_med_id in grin_meds:
                        g_val = g_med_id.item()
                        sub_mask = grin_mask & (curr_medium_ids == g_val)
                        if not sub_mask.any():
                            continue
                        field = self.grin_fields[g_val]
                        field.set_context(pol_vec_all[sub_mask], dop_all[sub_mask])
                        h_sub = ds_per_ray[sub_mask]
                        r_new, d_new, n_eval, h_new, err_new, accepted = \
                            adaptive_rk4_step_per_ray(
                                curr_origins[sub_mask], curr_dirs[sub_mask], field,
                                curr_wls[sub_mask], h_per_ray=h_sub,
                                tol=rk_tol, h_min=ds_min, h_max=ds_max)
                        s_old = curr_s_ref[sub_mask]
                        s_proj = torch.sum(s_old * d_new, dim=-1, keepdim=True)
                        s_candidate = s_old - s_proj * d_new
                        s_candidate = s_candidate / (
                                torch.norm(s_candidate, dim=-1, keepdim=True) + 1e-12)
                        stokes_candidate = rotate_stokes_reference(
                            curr_stokes[sub_mask], s_old, s_candidate, d_new)
                        acc_3d = accepted.unsqueeze(-1)
                        next_origins[sub_mask] = torch.where(
                            acc_3d, r_new, curr_origins[sub_mask])
                        next_dirs[sub_mask] = torch.where(
                            acc_3d, d_new, curr_dirs[sub_mask])
                        next_s_ref[sub_mask] = torch.where(
                            acc_3d, s_candidate, curr_s_ref[sub_mask])
                        next_p_ref[sub_mask] = torch.cross(
                            next_dirs[sub_mask], next_s_ref[sub_mask], dim=-1)
                        next_stokes[sub_mask] = torch.where(
                            acc_3d, stokes_candidate, curr_stokes[sub_mask])
                        n_eval_col = n_eval if n_eval.dim() == 2 else n_eval.unsqueeze(-1)
                        step_n1[sub_mask] = torch.where(
                            acc_3d, n_eval_col, step_n1[sub_mask])
                        step_n2[sub_mask] = torch.where(
                            acc_3d, n_eval_col, step_n2[sub_mask])
                        step_rk_err[sub_mask] = err_new
                        ds_per_ray[sub_mask] = h_new
                        field.clear_context()

            # ---------- 4. Surface interaction ----------
            mesh_hit_mask = active_mask & (~grin_mask) & (hit_face_ids >= 0)
            if mesh_hit_mask.any():
                hit_pts = curr_origins[mesh_hit_mask] + \
                          hit_dist[mesh_hit_mask].unsqueeze(-1) * \
                          curr_dirs[mesh_hit_mask]
                next_origins[mesh_hit_mask] = hit_pts

                normals = hit_normals[mesh_hit_mask]
                dirs_in = curr_dirs[mesh_hit_mask]
                stokes_in = curr_stokes[mesh_hit_mask]
                s_ref_in = curr_s_ref[mesh_hit_mask]

                cos_theta_i_raw = -torch.sum(dirs_in * normals, dim=-1,
                                             keepdim=True)
                is_front = cos_theta_i_raw > 0.0
                oriented_normals = torch.where(is_front, normals, -normals)
                cos_theta_i = torch.clamp(torch.abs(cos_theta_i_raw), 0.0, 1.0)

                faces_hit = hit_face_ids[mesh_hit_mask]
                mat_front = self.face_materials[faces_hit, 0]
                mat_back = self.face_materials[faces_hit, 1]
                n1_ids = torch.where(is_front.squeeze(-1), mat_front, mat_back)
                n2_ids = torch.where(is_front.squeeze(-1), mat_back, mat_front)

                wls_hit = curr_wls[mesh_hit_mask]
                cos_hit = cos_theta_i

                cross_plane = torch.cross(dirs_in, oriented_normals, dim=-1)
                norm_plane = torch.norm(cross_plane, dim=-1, keepdim=True)
                degenerate = norm_plane.squeeze(-1) < 1e-8
                s_plane = torch.where(
                    degenerate.unsqueeze(-1),
                    s_ref_in,
                    cross_plane / (norm_plane + 1e-12))

                stokes_plane = rotate_stokes_reference(
                    stokes_in, s_ref_in, s_plane, dirs_in)
                S1_plane = stokes_plane[:, 1:2]

                if self.all_simple:
                    n1_vals = self.n_table[n1_ids].unsqueeze(-1)
                    n2_vals = self.n_table[n2_ids].unsqueeze(-1)
                    custom_refl = self.r_table[n2_ids].unsqueeze(-1)
                    custom_diff = self.d_table[n2_ids].unsqueeze(-1)
                    custom_abs = self.a_table[n2_ids].unsqueeze(-1)
                else:
                    n1_vals = torch.zeros_like(wls_hit)
                    n2_vals = torch.zeros_like(wls_hit)
                    custom_refl = torch.zeros_like(wls_hit)
                    custom_diff = torch.zeros_like(wls_hit)
                    custom_abs = torch.zeros_like(wls_hit)
                    all_mats = torch.unique(torch.cat([n1_ids, n2_ids]))
                    for m_id in all_mats:
                        m_val = m_id.item()
                        fn_n = self.mat_evaluators[m_val]["n"]
                        fn_r = self.mat_evaluators[m_val]["reflectivity"]
                        fn_d = self.mat_evaluators[m_val]["diffuse"]
                        fn_a = self.mat_evaluators[m_val]["absorptivity"]
                        m1 = (n1_ids == m_val)
                        if m1.any():
                            n1_vals[m1] = fn_n(wls_hit[m1], cos_hit[m1])
                        m2 = (n2_ids == m_val)
                        if m2.any():
                            n2_vals[m2] = fn_n(wls_hit[m2], cos_hit[m2])
                            custom_refl[m2] = fn_r(wls_hit[m2], cos_hit[m2])
                            custom_diff[m2] = fn_d(wls_hit[m2], cos_hit[m2])
                            custom_abs[m2] = fn_a(wls_hit[m2], cos_hit[m2])

                n1_vals = torch.clamp(n1_vals, min=1.0)
                n2_vals = torch.clamp(n2_vals, min=1.0)

                eta = n1_vals / n2_vals
                sin2_t = torch.clamp(1.0 - cos_theta_i ** 2, min=0.0)
                sin2_tt = (eta ** 2) * sin2_t
                is_tir = sin2_tt > 1.0
                cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_tt, min=0.0))

                denom_s = n1_vals * cos_theta_i + n2_vals * cos_theta_t + 1e-12
                denom_p = n2_vals * cos_theta_i + n1_vals * cos_theta_t + 1e-12
                r_s = (n1_vals * cos_theta_i - n2_vals * cos_theta_t) / denom_s
                r_p = (n2_vals * cos_theta_i - n1_vals * cos_theta_t) / denom_p
                t_s = 2.0 * n1_vals * cos_theta_i / denom_s
                t_p = 2.0 * n1_vals * cos_theta_i / denom_p

                r_s = torch.where(is_tir, torch.ones_like(r_s), r_s)
                r_p = torch.where(is_tir, torch.ones_like(r_p), r_p)
                t_s = torch.where(is_tir, torch.zeros_like(t_s), t_s)
                t_p = torch.where(is_tir, torch.zeros_like(t_p), t_p)

                R_s = r_s ** 2
                R_p = r_p ** 2

                w_s = 0.5 * (1.0 + S1_plane)
                w_p = 0.5 * (1.0 - S1_plane)
                R_eff = w_s * R_s + w_p * R_p

                if surface_model == "fresnel":
                    specular_R = R_eff
                elif surface_model == "custom":
                    specular_R = custom_refl
                else:
                    specular_R = R_eff * custom_refl
                specular_R = torch.where(is_tir, torch.ones_like(specular_R),
                                         specular_R)

                P_absorb = torch.clamp(custom_abs, 0.0, 1.0)
                P_diffuse = (1.0 - P_absorb) * torch.clamp(custom_diff, 0.0, 1.0)
                P_specular = (1.0 - P_absorb - P_diffuse) * torch.clamp(
                    specular_R, 0.0, 1.0)
                P_refract = torch.clamp(
                    1.0 - P_absorb - P_diffuse - P_specular, min=0.0)

                total_P = P_absorb + P_diffuse + P_specular + P_refract
                total_safe = torch.clamp(total_P, min=1e-12)
                P_absorb = P_absorb / total_safe
                P_diffuse = P_diffuse / total_safe
                P_specular = P_specular / total_safe
                P_refract = P_refract / total_safe

                step_refl[mesh_hit_mask] = P_specular + P_diffuse
                step_trans[mesh_hit_mask] = P_refract
                step_abs[mesh_hit_mask] = P_absorb
                step_n1[mesh_hit_mask] = n1_vals
                step_n2[mesh_hit_mask] = n2_vals

                reflect_dirs = dirs_in + 2.0 * cos_theta_i * oriented_normals
                reflect_dirs = reflect_dirs / torch.norm(
                    reflect_dirs, dim=-1, keepdim=True)
                refract_dirs = eta * dirs_in + (
                        eta * cos_theta_i - cos_theta_t) * oriented_normals
                refract_dirs = refract_dirs / torch.norm(
                    refract_dirs, dim=-1, keepdim=True)

                num_hits = hit_pts.shape[0]
                r_sq = torch.rand(num_hits, 1, device=self.device)
                sin_theta_d = torch.sqrt(r_sq)
                cos_theta_d = torch.sqrt(1.0 - r_sq)
                phi_d = 2.0 * math.pi * torch.rand(num_hits, 1,
                                                   device=self.device)
                ref_vecs = torch.zeros_like(oriented_normals)
                ref_vecs[:, 2] = 1.0
                ref_vecs[torch.abs(oriented_normals[:, 2]) > 0.999] = \
                    torch.tensor([0.0, 1.0, 0.0], device=self.device)
                local_u = torch.cross(ref_vecs, oriented_normals, dim=-1)
                local_u = local_u / torch.norm(local_u, dim=-1, keepdim=True)
                local_v = torch.cross(oriented_normals, local_u, dim=-1)
                diffuse_dirs = (sin_theta_d * torch.cos(phi_d) * local_u +
                                sin_theta_d * torch.sin(phi_d) * local_v +
                                cos_theta_d * oriented_normals)
                diffuse_dirs = diffuse_dirs / torch.norm(
                    diffuse_dirs, dim=-1, keepdim=True)

                rand_val = torch.rand_like(P_absorb)
                is_abs = rand_val < P_absorb
                is_diff = (~is_abs) & (rand_val < (P_absorb + P_diffuse))
                is_spec = (~is_abs) & (~is_diff) & (
                        rand_val < (P_absorb + P_diffuse + P_specular))
                is_refr = ~(is_abs | is_diff | is_spec)

                branch_prob = (
                        P_absorb * is_abs.float() +
                        P_diffuse * is_diff.float() +
                        P_specular * is_spec.float() +
                        P_refract * is_refr.float()
                )
                new_weight = next_weight[mesh_hit_mask].unsqueeze(-1) * branch_prob
                new_weight = torch.where(is_abs, torch.zeros_like(new_weight),
                                         new_weight)
                next_weight[mesh_hit_mask] = new_weight.squeeze(-1)

                K = mesh_hit_mask.sum().item()
                rr_kill = torch.zeros(K, dtype=torch.bool, device=self.device)
                if rr_threshold > 0.0:
                    alive = ~is_abs.squeeze(-1)
                    cur_w = next_weight[mesh_hit_mask]
                    needs_rr = alive & (cur_w < rr_threshold)
                    surv_prob = (cur_w / rr_threshold).clamp(0.0, 1.0)
                    surv = torch.rand_like(cur_w) < surv_prob
                    boosted = torch.where(
                        surv,
                        torch.full_like(cur_w, rr_threshold),
                        cur_w)
                    next_weight[mesh_hit_mask] = torch.where(
                        needs_rr, boosted, cur_w)
                    rr_kill = needs_rr & (~surv)

                stokes_refl_plane = apply_jones_mueller(stokes_plane, r_s, r_p)
                stokes_refr_plane = apply_jones_mueller(stokes_plane, t_s, t_p)
                stokes_diff_plane = randomize_stokes_diffuse(stokes_plane)

                stokes_refl_new = rotate_stokes_reference(
                    stokes_refl_plane, s_plane, s_plane, reflect_dirs)
                stokes_refr_new = rotate_stokes_reference(
                    stokes_refr_plane, s_plane, s_plane, refract_dirs)

                new_d = torch.where(is_diff, diffuse_dirs,
                                    torch.where(is_spec, reflect_dirs,
                                                refract_dirs))
                s_diff_default = _choose_perpendicular(diffuse_dirs)
                new_s_ref = torch.where(is_diff, s_diff_default,
                                        torch.where(is_spec, s_plane, s_plane))
                new_stokes = torch.where(is_diff, stokes_diff_plane,
                                         torch.where(is_spec, stokes_refl_new,
                                                     stokes_refr_new))

                next_dirs[mesh_hit_mask] = new_d
                next_s_ref[mesh_hit_mask] = new_s_ref
                next_p_ref[mesh_hit_mask] = torch.cross(new_d, new_s_ref, dim=-1)
                next_stokes[mesh_hit_mask] = new_stokes

                is_reflected = is_spec | is_diff
                curr_medium_ids[mesh_hit_mask] = torch.where(
                    is_reflected.squeeze(-1), n1_ids, n2_ids)
                prev_hit_faces[mesh_hit_mask] = faces_hit

                if is_abs.any():
                    absorbed_idx = torch.where(mesh_hit_mask)[0][
                        is_abs.squeeze(-1)]
                    active_mask[absorbed_idx] = False

                if rr_kill.any():
                    rr_idx = torch.where(mesh_hit_mask)[0][rr_kill]
                    active_mask[rr_idx] = False

            # ---------- 5. Escape ----------
            no_hit_mask = active_mask & (~grin_mask) & (hit_face_ids < 0)
            if no_hit_mask.any():
                next_origins[no_hit_mask] = curr_origins[no_hit_mask] + \
                                            curr_dirs[no_hit_mask] * 1.0
                active_mask[no_hit_mask] = False

            # ---------- 6. Update ----------
            curr_origins = next_origins
            curr_dirs = next_dirs
            curr_stokes = next_stokes
            curr_s_ref = next_s_ref
            curr_p_ref = next_p_ref
            curr_weight = next_weight

            # ---------- 7. Record ----------
            hist_pos.append(curr_origins.clone())
            hist_stokes.append(curr_stokes.clone())
            hist_active.append(active_mask.clone())
            hist_refl.append(step_refl)
            hist_trans.append(step_trans)
            hist_abs.append(step_abs)
            hist_n1.append(step_n1)
            hist_n2.append(step_n2)
            hist_pol_deg.append(
                0.5 * torch.atan2(curr_stokes[:, 2:3], curr_stokes[:, 1:2])
                * (180.0 / math.pi))
            hist_energy.append(curr_weight.clone())
            hist_rk_err.append(step_rk_err)
            hist_ds.append(ds_per_ray.clone())
            hist_dop.append(degree_of_polarization(curr_stokes).clone())

            step += 1

        # Energy conservation check
        energy_hist = torch.stack(hist_energy, dim=1)

        in_range = bool(torch.all(
            (energy_hist >= -1e-6) & (energy_hist <= 1.0 + 1e-6)).item())

        if energy_hist.shape[1] >= 2:
            diffs = energy_hist[:, 1:] - energy_hist[:, :-1]
            monotonic = bool(torch.all(diffs <= 1e-6).item())
        else:
            monotonic = True

        final_active = active_mask
        final_energy = energy_hist[:, -1]
        inactive_nonneg = bool(torch.all(
            (final_active) | (final_energy >= -1e-6)).item())

        energy_ok = (in_range and monotonic and inactive_nonneg)

        return {
            "positions": torch.stack(hist_pos, dim=1),
            "stokes": torch.stack(hist_stokes, dim=1),
            "active": torch.stack(hist_active, dim=1),
            "wavelengths": curr_wls,
            "reflectivity": torch.stack(hist_refl, dim=1),
            "transmissivity": torch.stack(hist_trans, dim=1),
            "absorptivity": torch.stack(hist_abs, dim=1),
            "n1": torch.stack(hist_n1, dim=1),
            "n2": torch.stack(hist_n2, dim=1),
            "polarization_deg": torch.stack(hist_pol_deg, dim=1),
            "dop": torch.stack(hist_dop, dim=1),
            "energy": energy_hist,
            "rk_error": torch.stack(hist_rk_err, dim=1),
            "ds_history": torch.stack(hist_ds, dim=1),
            "energy_conserved": energy_ok,
        }