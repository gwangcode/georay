"""
quick_check_stepfiles.py
========================
Verify stepmesh.glue() -> GeoRay pipeline with real STEP files.
"""
import numpy as np
import torch
from collections import Counter

from stepmesh import glue
from georay.stepmesh_adapter import (
    glue_to_georay, summarize_glue_output, validate_georay_input,
)
from georay import AdvancedRayTracerPBRTGPU, generate_rays


STEP_FILES = [
    "examples/cone.step",
    "examples/cube.step",
    "examples/cylinder.step",
]


def main():
    # ============================================================
    # 1. glue
    # ============================================================
    print("=" * 70)
    print("Step 1: Running stepmesh.glue() ...")
    print("=" * 70)

    vertices, faces, mapping_table, file_to_ids_dict, _ = glue(
        step_paths=STEP_FILES,
        min_size=0.5,
        max_size=2.0,
        return_internal_surfaces=True,
        return_each_tet_as_body=False,
    )
    summarize_glue_output(vertices, faces, mapping_table, file_to_ids_dict)

    # ============================================================
    # 2. vol_id -> material_id
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 2: Building vol_id -> material_id map ...")
    print("=" * 70)

    vol_to_material = {}
    materials_cfg = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }
    mat_id = 1
    for fname, vids in file_to_ids_dict.items():
        n_val = 1.5 + 0.2 * (mat_id - 1)
        materials_cfg[mat_id] = {
            "n": float(n_val), "reflectivity": 0.0,
            "diffuse": 0.0, "absorptivity": 0.0,
        }
        for vid in vids:
            vol_to_material[vid] = mat_id
        print(f"  {fname} -> material {mat_id} (n={n_val:.2f})")
        mat_id += 1

    # ============================================================
    # 3. Convert
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 3: Converting to GeoRay inputs ...")
    print("=" * 70)

    verts_t, faces_t, face_mats_t = glue_to_georay(
        vertices, faces, mapping_table,
        vol_to_material=vol_to_material,
        outside_material_id=0,
        device=torch.device("cpu"),
        verbose=True,
    )
    validate_georay_input(verts_t, faces_t, face_mats_t)

    # ============================================================
    # 4. Grid rays covering the entire bounding box
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 4: Tracing a grid of rays along +z ...")
    print("=" * 70)

    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    print(f"  Bounding box: min={bmin.tolist()}, max={bmax.tolist()}")

    rays = generate_rays(
        source_type="grid", num_rays=100,       # 10x10
        origin=(float((bmin[0]+bmax[0])/2),
                float((bmin[1]+bmax[1])/2),
                float(bmin[2] - 1.0)),
        direction=(0.0, 0.0, 1.0),
        bounds=(float(bmin[0] - 0.1), float(bmax[0] + 0.1),
                float(bmin[1] - 0.1), float(bmax[1] + 0.1)),
        polarization="random",
        wavelength_nm=550.0,
        device=torch.device("cpu"),
    )

    tracer = AdvancedRayTracerPBRTGPU(
        verts_t, faces_t, face_mats_t, materials_cfg, torch.device("cpu"))

    out = tracer.trace(
        rays, max_steps=2000,
        ds_init=0.05, ds_min=1e-3, ds_max=0.2,
        rk_tol=1e-4, fresnel=False,
    )

    # ============================================================
    # 5. Statistics across all rays
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 5: Material transitions across all rays")
    print("=" * 70)

    N = out["n1"].shape[0]
    n1_all = out["n1"][:, :, 0].cpu().numpy()
    n2_all = out["n2"][:, :, 0].cpu().numpy()

    pair_counter = Counter()
    n_rays_hit = 0
    for i in range(N):
        if (n1_all[i] > 1.0).any() or (n2_all[i] > 1.0).any():
            n_rays_hit += 1
        for step in range(n1_all.shape[1]):
            n1 = float(n1_all[i, step])
            n2 = float(n2_all[i, step])
            if n1 > 1.0 or n2 > 1.0:
                pair_counter[(round(n1, 2), round(n2, 2))] += 1

    print(f"  Rays that hit any object: {n_rays_hit} / {N}")
    print(f"  Material transitions (n1 -> n2), by frequency:")
    for (n1, n2), count in sorted(pair_counter.items(), key=lambda x: -x[1]):
        print(f"    {n1:.2f} -> {n2:.2f}   ({count} times)")

    # ============================================================
    # 6. Judgement
    # ============================================================
    print("\n" + "=" * 70)
    print("Judgement")
    print("=" * 70)

    if n_rays_hit == 0:
        print("⚠️  No rays hit any object.")
    else:
        entering_cone = pair_counter.get((1.00, 1.50), 0)
        entering_cube = pair_counter.get((1.00, 1.70), 0)
        entering_cyl  = pair_counter.get((1.00, 1.90), 0)
        reversed_cone = pair_counter.get((1.50, 1.00), 0)
        reversed_cube = pair_counter.get((1.70, 1.00), 0)
        reversed_cyl  = pair_counter.get((1.90, 1.00), 0)

        total_entering = entering_cone + entering_cube + entering_cyl
        total_reversed = reversed_cone + reversed_cube + reversed_cyl

        print(f"  'air -> material' (entering): {total_entering}")
        print(f"  'material -> air' (exiting):  {total_reversed}")
        print()
        if total_entering > 0 and total_entering >= total_reversed:
            print("  ✅ Material order is CORRECT (face_mats = [mat_back, mat_front])")
        elif total_reversed > 0 and total_reversed > total_entering:
            print("  ❌ Material order is REVERSED.")
            print("     Fix: in stepmesh_adapter.py, change to")
            print("     face_mats = np.stack([mat_front, mat_back], axis=1)")


if __name__ == "__main__":
    main()