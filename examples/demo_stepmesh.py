"""
Demo: Trace rays through a multi-body STEP assembly remeshed with stepmesh.glue().

Prerequisites:
    pip install stepmesh
    Three STEP files: cube.step, cone.step, cylinder.step (placed in examples/)
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
    # 1. Locate STEP files
    # ---------------------------------------------------------------
    from stepmesh import glue

    examples_dir = os.path.dirname(os.path.abspath(__file__))
    step_files = [
        os.path.join(examples_dir, "cube.step"),
        os.path.join(examples_dir, "cone.step"),
        os.path.join(examples_dir, "cylinder.step"),
    ]

    for f in step_files:
        assert os.path.exists(f), "STEP file not found: {}".format(f)
        print(f"  [OK] {os.path.basename(f)}")

    # ---------------------------------------------------------------
    # 2. Call stepmesh.glue()
    # ---------------------------------------------------------------
    print("\nCalling stepmesh.glue()...")

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
        raise RuntimeError("glue() returned empty geometry; check STEP files")

    summarize_glue_output(vertices, faces, mapping_table, file_to_ids_dict)

    # ---------------------------------------------------------------
    # 3. Build volume ID -> material ID mapping
    # ---------------------------------------------------------------
    all_vols = sorted(set(mapping_table[:, 0].tolist()) |
                      set(mapping_table[:, 1].tolist()))
    print(f"\nAll volume IDs: {all_vols}")

    vol_to_material = {0: 0}
    object_materials = [1, 2, 3]
    non_zero_vols = [v for v in all_vols if v != 0]
    for i, v in enumerate(non_zero_vols):
        vol_to_material[v] = object_materials[i % len(object_materials)]

    print(f"vol_to_material = {vol_to_material}")

    # ---------------------------------------------------------------
    # 4. Convert to GeoRay inputs
    # ---------------------------------------------------------------
    print("\nConverting to GeoRay tensors...")
    verts, faces_t, face_mats = glue_to_georay(
        vertices, faces, mapping_table,
        vol_to_material=vol_to_material,
        outside_material_id=0,
        device=device,
    )
    validate_georay_input(verts, faces_t, face_mats)

    # ---------------------------------------------------------------
    # 5. Material configuration
    # ---------------------------------------------------------------
    materials_cfg = {
        0: {"n": 1.00, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.50, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        2: {"n": 1.77, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        3: {"n": 1.33, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # ---------------------------------------------------------------
    # 6. Generate rays
    # ---------------------------------------------------------------
    v_np = verts.detach().cpu().numpy()
    bmin = v_np.min(axis=0)
    bmax = v_np.max(axis=0)
    center = 0.5 * (bmin + bmax)
    size = bmax - bmin

    print(f"\nScene bbox: min={np.round(bmin, 3).tolist()}, "
          f"max={np.round(bmax, 3).tolist()}")
    print(f"Scene size: {np.round(size, 3).tolist()}")

    # Use a perfect square for the number of rays (grid source requirement)
    N_RAYS = 25    # 5x5 grid

    # Rays are incident from the -z direction
    z_start = float(bmin[2] - 0.5 * size[2])

    # Key fix: `bounds` is an OFFSET relative to origin, not absolute coordinates
    span_x = 0.4 * size[0]   # +/- 40% of size, covers the objects
    span_y = 0.4 * size[1]

    rays = generate_rays(
        source_type="grid", num_rays=N_RAYS,
        origin=(float(center[0]), float(center[1]), z_start),
        direction=(0.0, 0.0, 1.0),
        # Pass offsets, not absolute coordinates
        bounds=(-span_x, span_x, -span_y, span_y),
        polarization="random",
        wavelength_nm=550.0,
        device=device,
    )

    rays["medium_ids"] = torch.zeros(rays["origins"].shape[0],
                                       dtype=torch.long, device=device)

    print(f"Generated {rays['origins'].shape[0]} rays")
    print(f"  z start:  {z_start:.3f}")
    print(f"  x range:  [{center[0] - span_x:.3f}, {center[0] + span_x:.3f}]")
    print(f"  y range:  [{center[1] - span_y:.3f}, {center[1] + span_y:.3f}]")

    # Check whether the rays cover the objects
    print(f"  Object x range: [{bmin[0]:.3f}, {bmax[0]:.3f}]")
    print(f"  Object y range: [{bmin[1]:.3f}, {bmax[1]:.3f}]")

    # ---------------------------------------------------------------
    # 7. Trace
    # ---------------------------------------------------------------
    print("\nTracing...")
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

    # Full diagnostics
    print("\n  Per-ray hit status:")
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
    print(f"    Hit objects:   {n_hit} / {pos.shape[0]}")
    print(f"    Missed:        {n_miss} / {pos.shape[0]}")

    # Direction change diagnostics (first 5 rays)
    print("\nDirection change for the first 5 rays (refraction check):")
    for i in range(min(5, pos.shape[0])):
        valid_len = int(active[i].sum()) + 1
        if valid_len < 3:
            print(f"  Ray {i}: missed ({valid_len} steps)")
            continue
        d_before = pos[i, 1] - pos[i, 0]
        d_before = d_before / (np.linalg.norm(d_before) + 1e-12)
        d_after = pos[i, 2] - pos[i, 1]
        d_after = d_after / (np.linalg.norm(d_after) + 1e-12)
        cos_angle = np.clip(np.dot(d_before, d_after), -1.0, 1.0)
        angle_change = np.degrees(np.arccos(cos_angle))
        print(f"  Ray {i}: direction change {angle_change:.2f} deg")

    # n1 / n2 history
    for i in range(min(3, pos.shape[0])):
        n1_hist = out["n1"][i, :, 0].cpu().tolist()
        n2_hist = out["n2"][i, :, 0].cpu().tolist()
        print(f"  Ray {i} n1 first 5 steps: {[round(v, 3) for v in n1_hist[:5]]}")
        print(f"  Ray {i} n2 first 5 steps: {[round(v, 3) for v in n2_hist[:5]]}")

    # ---------------------------------------------------------------
    # 8. Export
    # ---------------------------------------------------------------
    output_npz = os.path.join(examples_dir, "stepmesh_trace.npz")
    output_stl = os.path.join(examples_dir, "stepmesh_trace.stl")

    save_trace_output_to_npz(out, output_npz)

    # Ray radius reduced from 0.03 to 0.005 (6x thinner)
    ray_radius = 0.005 * float(size.max())
    print(f"\nRay radius: {ray_radius:.4f} (0.005 x bbox_max)")
    export_ray_history_to_stl(out, output_stl,
                               radius=ray_radius, sides=8)

    # Scene STL
    import trimesh
    scene_stl = os.path.join(examples_dir, "scene.stl")
    scene_mesh = trimesh.Trimesh(
        vertices=verts.detach().cpu().numpy(),
        faces=faces_t.detach().cpu().numpy(),
    )
    scene_mesh.export(scene_stl)

    print(f"\nNPZ:   {output_npz}")
    print(f"STL:   {output_stl}")
    print(f"Scene: {scene_stl}")


if __name__ == "__main__":
    main()
