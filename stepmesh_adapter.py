"""
georay/stepmesh_adapter.py
==========================
Convert stepmesh.glue() output into GeoRay-compatible inputs.
"""
import numpy as np
import torch


def glue_to_georay(
    vertices,
    faces,
    mapping_table,
    vol_to_material,
    outside_material_id=0,
    device=None,
    remove_degenerate=True,
    verbose=True,
):
    """
    Convert stepmesh.glue() output to GeoRay inputs.

    Args:
        vertices: (N, 3) float64 — node coordinates.
        faces: (M, 3) int64 — triangle vertex indices.
        mapping_table: (M, 3) int32 — [pos_vol, neg_vol, local_id].
        vol_to_material: dict {vol_id: material_id}. Any volume ID not in
            this dict is treated as air (outside_material_id).
        outside_material_id: int — material ID for the outside (default 0).
        device: torch.device or None.
        remove_degenerate: remove zero-area triangles and compact vertices.
        verbose: print summary.

    Returns:
        (verts_t, faces_t, face_mats_t) — three torch tensors ready to be
        passed directly to AdvancedRayTracerPBRTGPU.
        If all triangles are degenerate, returns empty tensors with shapes
        (0, 3), (0, 3), (0, 2).
    """
    # ---------- Input validation ----------
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mapping_table = np.asarray(mapping_table, dtype=np.int64)

    assert vertices.ndim == 2 and vertices.shape[1] == 3, \
        f"vertices must be (N, 3), got {vertices.shape}"
    assert faces.ndim == 2 and faces.shape[1] == 3, \
        f"faces must be (M, 3), got {faces.shape}"
    assert mapping_table.shape == (faces.shape[0], 3), \
        f"mapping_table shape {mapping_table.shape} != ({faces.shape[0]}, 3)"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Inline degenerate triangle removal ----------
    if remove_degenerate and faces.shape[0] > 0:
        tris = vertices[faces]                          # (M, 3, 3)
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=-1)   # (M,)
        keep = areas > 1e-12                             # (M,) bool

        n_removed = int((~keep).sum())
        if n_removed > 0:
            faces = faces[keep]
            mapping_table = mapping_table[keep]

            if faces.shape[0] > 0:
                # Reindex vertices to eliminate unused nodes
                used = np.unique(faces)                  # sorted unique vertex IDs
                remap = -np.ones(vertices.shape[0], dtype=np.int64)
                remap[used] = np.arange(used.shape[0], dtype=np.int64)
                faces = remap[faces]
                vertices = vertices[used]
            else:
                # All triangles were degenerate
                vertices = np.zeros((0, 3), dtype=np.float64)

            if verbose:
                print(f"  Removed {n_removed} degenerate triangles "
                      f"(now {faces.shape[0]} faces, {vertices.shape[0]} verts)")

    M = faces.shape[0]
    if M == 0:
        # All triangles degenerate → return empty tensors gracefully
        if verbose:
            print("  Warning: all triangles were degenerate; returning empty mesh.")
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, 3), dtype=torch.long, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
        )

    # ---------- Build face_materials ----------
    # glue convention:  normal points OUTWARD (away from pos_vol)
    #   face_materials[:, 0] = material on the BACK side  = mat(neg_vol)
    #   face_materials[:, 1] = material on the FRONT side = mat(pos_vol)
    # If the quick_check test shows n1/n2 reversed, swap the two lines below.
    pos_vols = mapping_table[:, 0]
    neg_vols = mapping_table[:, 1]

    mat_back = np.full(M, outside_material_id, dtype=np.int64)
    mat_front = np.full(M, outside_material_id, dtype=np.int64)

    for vol_id, mat_id in vol_to_material.items():
        mat_front[pos_vols == vol_id] = mat_id
        mat_back[neg_vols == vol_id] = mat_id

    face_mats = np.stack([mat_back, mat_front], axis=1)   # (M, 2)

    # ---------- Convert to torch ----------
    verts_t = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces, dtype=torch.long, device=device)
    face_mats_t = torch.tensor(face_mats, dtype=torch.long, device=device)

    if verbose:
        unique_mats = sorted(set(face_mats.flatten().tolist()))
        print(f"  glue_to_georay: {verts_t.shape[0]} verts, "
              f"{faces_t.shape[0]} faces, material IDs = {unique_mats}")

    return verts_t, faces_t, face_mats_t


def summarize_glue_output(vertices, faces, mapping_table,
                          file_to_ids_dict=None):
    """Print a summary of glue() output for inspection."""
    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]
    n_outer = int((mapping_table[:, 1] == 0).sum()) if n_faces > 0 else 0
    n_internal = int((mapping_table[:, 1] > 0).sum()) if n_faces > 0 else 0

    print("=== stepmesh.glue() output summary ===")
    print(f"  Vertices:          {n_verts}")
    print(f"  Faces:             {n_faces}")
    print(f"    - Outer surface: {n_outer}")
    print(f"    - Internal face: {n_internal}")

    if n_faces > 0:
        print(f"  Physical vols:     "
              f"{sorted(set(mapping_table[:, 0].tolist()))}")

    if file_to_ids_dict:
        print("  File -> vol map:")
        for fname, vids in file_to_ids_dict.items():
            print(f"    {fname}: {vids}")

    # Check watertightness: every edge should appear exactly twice
    if n_faces > 0:
        edge_count = {}
        for f in faces:
            for e in ((int(f[0]), int(f[1])),
                      (int(f[1]), int(f[2])),
                      (int(f[2]), int(f[0]))):
                key = (min(e), max(e))
                edge_count[key] = edge_count.get(key, 0) + 1

        non_manifold = sum(1 for c in edge_count.values() if c != 2)
        if non_manifold == 0:
            status = "watertight OK"
        else:
            status = f"NOT watertight ({non_manifold} edges with count != 2)"
        print(f"  Watertightness:    {status}")


def validate_georay_input(verts, faces, face_mats, verbose=True):
    """Sanity-check the converted tensors before passing to the tracer."""
    M = faces.shape[0]
    assert face_mats.shape == (M, 2), \
        f"face_mats must be (M, 2), got {tuple(face_mats.shape)}"
    assert verts.shape[1] == 3, f"verts must be (N, 3), got {tuple(verts.shape)}"
    assert faces.shape[1] == 3, f"faces must be (M, 3), got {tuple(faces.shape)}"

    if M > 0:
        assert faces.min().item() >= 0, "negative face index"
        assert faces.max().item() < verts.shape[0], "face index out of range"

    if verbose:
        unique_mats = torch.unique(face_mats).cpu().tolist() if M > 0 else []
        print(f"  Validated: {M} faces, materials = {unique_mats}")
    return True