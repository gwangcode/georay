"""Demo 3: Polarization rotation at a glass interface (s vs p)."""
import torch
import trimesh

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Flat slab (2 triangles)
    verts = torch.tensor([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    face_mats = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)

    materials_cfg = {
        0: {"n": 1.0, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 1.0, "diffuse": 0.0, "absorptivity": 0.0},
    }

    # Normal-incidence ray with s-polarization
    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0),
        bounds=(-0.001, 0.001, -0.001, 0.001),
        polarization="s", wavelength_nm=550.0, device=device,
    )

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, face_mats, materials_cfg, device)
    out = tracer.trace(rays, max_steps=4, fresnel=True)

    print(f"Polarization (deg) history: {out['polarization_deg'][0, :, 0].tolist()}")
    print(f"Reflectivity history:       {out['reflectivity'][0, :, 0].tolist()}")
    print(f"Transmissivity history:     {out['transmissivity'][0, :, 0].tolist()}")
    print(f"Energy conserved: {out['energy_conserved']}")


if __name__ == "__main__":
    main()