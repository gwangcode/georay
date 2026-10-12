"""Unit tests for the stepmesh -> GeoRay adapter."""
import numpy as np
import torch

from georay.stepmesh_adapter import glue_to_georay


def test_outer_surface_only():
    """A single triangle with pos_vol=1, neg_vol=0."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mapping_table = np.array([[1, 0, 0]], dtype=np.int32)

    verts, f, fm = glue_to_georay(
        vertices, faces, mapping_table,
        vol_to_material={1: 5},
        outside_material_id=0,
        device=torch.device("cpu"),
        verbose=False,
    )
    # Order should be [mat_neg, mat_pos] = [0, 5]
    assert fm[0, 0].item() == 0, f"expected 0, got {fm[0, 0].item()}"
    assert fm[0, 1].item() == 5, f"expected 5, got {fm[0, 1].item()}"


def test_internal_interface():
    """A shared face between vol 1 (mat 10) and vol 2 (mat 20)."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mapping_table = np.array([[1, 2, 0]], dtype=np.int32)

    verts, f, fm = glue_to_georay(
        vertices, faces, mapping_table,
        vol_to_material={1: 10, 2: 20},
        outside_material_id=0,
        device=torch.device("cpu"),
        verbose=False,
    )
    # Order should be [mat_neg, mat_pos] = [20, 10]
    assert fm[0, 0].item() == 20, f"expected 20, got {fm[0, 0].item()}"
    assert fm[0, 1].item() == 10, f"expected 10, got {fm[0, 1].item()}"


def test_degenerate_removal():
    """Degenerate triangle should be removed."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)   # collinear -> zero area
    mapping_table = np.array([[1, 0, 0]], dtype=np.int32)

    verts, f, fm = glue_to_georay(
        vertices, faces, mapping_table,
        vol_to_material={1: 5},
        device=torch.device("cpu"),
        verbose=False,
    )
    assert f.shape[0] == 0, "degenerate triangle should be removed"