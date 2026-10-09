"""Unit tests for BVH ray-mesh intersection."""
import torch
import trimesh

from georay.bvh import BVHMeshIntersector


def test_bvh_hit_cube():
    device = torch.device("cpu")
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)

    intersector = BVHMeshIntersector(verts, faces, device)

    origin = torch.tensor([[0.0, 0.0, -3.0]], device=device)
    direction = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    hit_dist, hit_face, _ = intersector.intersect(origin, direction)
    assert hit_face[0].item() >= 0, "Should hit the cube"
    assert 1.9 < hit_dist[0].item() < 2.1, f"Distance should be ~2, got {hit_dist[0].item()}"


def test_bvh_miss():
    device = torch.device("cpu")
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)

    intersector = BVHMeshIntersector(verts, faces, device)
    origin = torch.tensor([[10.0, 10.0, -3.0]], device=device)
    direction = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    hit_dist, hit_face, _ = intersector.intersect(origin, direction)
    assert hit_face[0].item() == -1, "Should miss"