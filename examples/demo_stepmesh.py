"""
Demo: Trace rays through a multi-body STEP assembly remeshed with stepmesh.glue().

Prerequisites:
    pip install stepmesh   # or your local glue() implementation
    Two or more STEP files in the current directory.
"""
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
    # 1. Run stepmesh.glue() to obtain the remeshed geometry
    # ---------------------------------------------------------------
    from stepmesh import glue  # noqa

    step_files = ["part_a.step", "part_b.step"]   # <-- your STEP files

    vertices, faces, mapping_table, file_to_ids_dict, _ = glue(
        step_paths=step_files,
        min_size=0.5,
        max_size=2.0,
        return_internal_surfaces=True,      # ★ MUST be True
        return_each_tet_as_body=False,
    )

    summarize_glue_output(vertices, faces, mapping_table, file_to_ids_dict)

    # ---------------------------------------------------------------
    # 2. Map physical volume IDs -> material IDs
    # ---------------------------------------------------------------
    # Example: each STEP file gets its own material
    vol_to_material = {}
    for fname, vids in file_to_ids_dict.items():
        if "part_a" in fname:
            mat_id = 1     # glass A, n=1.5
        elif "part_b" in fname:
            mat_id = 2     # glass B, n=1.7
        else:
            mat_id = 1
        for vid in vids:
            vol_to_material[vid] = mat_id

    print(f"\nvol_to_material = {vol_to_material}")

    # ---------------------------------------------------------------
    # 3. Convert to GeoRay inputs
    # ---------------------------------------------------------------
    print("\nConverting glue() output to GeoRay inputs ...")
    verts, faces_t, face_mats = glue_to_georay(
        vertices, faces, mapping_table,
        vol_to_material=vol_to_material,
        outside_material_id=0,
        device=device,
    )
    validate_georay_input(verts, faces_t, face_mats)

    # ---------------------------------------------------------------
    # 4. Materials config
    # ---------------------------------------------------------------
    materials_cfg = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        2: {"n": 1.7, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # ---------------------------------------------------------------
    # 5. Generate rays (auto bounding box)
    # ---------------------------------------------------------------
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    center = (bmin + bmax) / 2
    span = float((bmax - bmin).max())

    rays = generate_rays(
        source_type="grid", num_rays=64,
        origin=(float(center[0]), float(center[1]), float(bmin[2] - 2.0 * span)),
        direction=(0.0, 0.0, 1.0),
        bounds=(float(center[0] - span), float(center[0] + span),
                float(center[1] - span), float(center[1] + span)),
        polarization="random",
        wavelength_nm=550.0,
        device=device,
    )

    # ---------------------------------------------------------------
    # 6. Trace
    # ---------------------------------------------------------------
    tracer = AdvancedRayTracerPBRTGPU(
        verts, faces_t, face_mats, materials_cfg, device)

    out = tracer.trace(
        rays, max_steps=1000,
        ds_init=0.02, ds_min=1e-3, ds_max=0.1,
        rk_tol=1e-4, fresnel=True,
    )

    print(f"\n=== Trace result ===")
    print(f"  Energy conserved: {out['energy_conserved']}")
    print(f"  Positions shape:  {out['positions'].shape}")
    print(f"  Stokes shape:     {out['stokes'].shape}")

    # Show n1/n2 history (verify materials are correct)
    n1_hist = out["n1"][0, :, 0].cpu().tolist()
    n2_hist = out["n2"][0, :, 0].cpu().tolist()
    print(f"  Ray 0 n1 history: {n1_hist[:10]}")
    print(f"  Ray 0 n2 history: {n2_hist[:10]}")

    # ---------------------------------------------------------------
    # 7. Export
    # ---------------------------------------------------------------
    save_trace_output_to_npz(out, "stepmesh_trace.npz")
    export_ray_history_to_stl(out, "stepmesh_trace.stl")


if __name__ == "__main__":
    main()