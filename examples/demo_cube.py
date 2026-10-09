"""Demo 1: Rays through a cube with Fresnel reflection/refraction."""
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Cube
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    verts = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.long, device=device)
    face_mats = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    face_mats[:, 0] = 0
    face_mats[:, 1] = 1

    # Materials
    materials_cfg = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # Rays
    rays = generate_rays(
        source_type="grid", num_rays=16,
        origin=(0.0, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.5, 0.5, -0.5, 0.5),
        polarization="s", wavelength_nm=550.0, device=device,
    )

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials_cfg, device)
    out = tracer.trace(rays, max_steps=40, fresnel=True)

    print(f"Rays traced: {rays['origins'].shape[0]}")
    print(f"Energy conserved: {out['energy_conserved']}")
    print(f"Output shape: {out['positions'].shape}")


if __name__ == "__main__":
    main()