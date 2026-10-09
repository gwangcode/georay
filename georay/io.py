"""Export utilities for ray tracing results."""
import numpy as np
import torch
import trimesh


def save_trace_output_to_npz(trace_output, filename="trace_output.npz"):
    """Save trace output dict to compressed NPZ."""
    data_dict = {k: v.cpu().numpy() for k, v in trace_output.items()
                 if isinstance(v, torch.Tensor)}
    np.savez_compressed(filename, **data_dict)
    print(f"Saved NPZ: {filename}")


def export_ray_history_to_stl(trace_output, filename="ray_tracks.stl",
                              radius=0.02, sides=6):
    """Export ray tracks as a 3D STL mesh of cylinders."""
    positions = trace_output["positions"].cpu().numpy()
    active = trace_output["active"].cpu().numpy()
    N, S, _ = positions.shape
    meshes = []
    for i in range(N):
        valid_len = min(int(np.sum(active[i])) + 1, S)
        if valid_len < 2:
            continue
        pts = positions[i, :valid_len]
        for j in range(valid_len - 1):
            p0, p1 = pts[j], pts[j + 1]
            if np.allclose(p0, p1):
                continue
            seg = trimesh.creation.cylinder(radius=radius,
                                            segment=np.array([p0, p1]),
                                            sections=sides)
            meshes.append(seg)
    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.export(filename)
        print(f"Exported STL: {filename}")
    else:
        print("Warning: no valid ray tracks to export")