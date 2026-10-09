"""Light source generators."""
import math
import torch

from .polarization import _choose_perpendicular


def generate_rays(source_type="grid", num_rays=100, origin=(0, 0, 0),
                  direction=(0, 0, 1), radius=1.0, angle_deg=10.0,
                  bounds=(-1, 1, -1, 1), mesh=None, polarization="random",
                  wavelength_nm=550.0, device=torch.device("cpu")):
    """Generate rays of various source types.

    Args:
        source_type: "grid", "random", "point_source", "area_source", or "mesh_source".
        num_rays: number of rays (rounded to square for grid).
        origin: source center (x, y, z).
        direction: main propagation direction.
        radius: radius for random source.
        angle_deg: cone half-angle for point_source.
        bounds: (xmin, xmax, ymin, ymax) for grid/area_source.
        mesh: trimesh.Trimesh for mesh_source.
        polarization: "s", "p", "random", or angle in degrees.
        wavelength_nm: wavelength in nanometers.
        device: torch device.

    Returns:
        dict with keys: origins, dirs, wavelengths, stokes, s_ref, p_ref, ray_ids.
    """
    main_dir = torch.tensor(direction, dtype=torch.float32, device=device)
    main_dir = main_dir / torch.norm(main_dir)
    center_org = torch.tensor(origin, dtype=torch.float32, device=device)

    ref_vec = torch.tensor([0.0, 0.0, 1.0], device=device) if torch.abs(main_dir[2]) < 0.999 \
        else torch.tensor([0.0, 1.0, 0.0], device=device)
    u_vec = torch.cross(ref_vec, main_dir, dim=-1)
    u_vec = u_vec / torch.norm(u_vec)
    v_vec = torch.cross(main_dir, u_vec, dim=-1)

    if source_type == "grid":
        side_len = int(math.sqrt(num_rays))
        num_rays = side_len * side_len
        x_coords = torch.linspace(bounds[0], bounds[1], side_len, device=device)
        y_coords = torch.linspace(bounds[2], bounds[3], side_len, device=device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        origins = center_org + grid_x.reshape(-1, 1) * u_vec + grid_y.reshape(-1, 1) * v_vec
        dirs = main_dir.unsqueeze(0).expand(num_rays, 3)
    elif source_type == "random":
        r_rand = radius * torch.sqrt(torch.rand(num_rays, 1, device=device))
        theta_rand = 2 * math.pi * torch.rand(num_rays, 1, device=device)
        origins = center_org + (r_rand * torch.cos(theta_rand)) * u_vec \
                  + (r_rand * torch.sin(theta_rand)) * v_vec
        dirs = main_dir.unsqueeze(0).expand(num_rays, 3)
    elif source_type == "point_source":
        origins = center_org.unsqueeze(0).expand(num_rays, 3)
        max_theta = math.radians(angle_deg)
        cos_theta = 1 - torch.rand(num_rays, 1, device=device) * (1 - math.cos(max_theta))
        sin_theta = torch.sqrt(1 - cos_theta ** 2)
        phi = 2 * math.pi * torch.rand(num_rays, 1, device=device)
        dirs = sin_theta * torch.cos(phi) * u_vec \
               + sin_theta * torch.sin(phi) * v_vec + cos_theta * main_dir
    elif source_type == "area_source":
        rand_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(num_rays, 1, device=device)
        rand_y = bounds[2] + (bounds[3] - bounds[2]) * torch.rand(num_rays, 1, device=device)
        origins = center_org + rand_x * u_vec + rand_y * v_vec
        r_sq = torch.rand(num_rays, 1, device=device)
        sin_theta = torch.sqrt(r_sq)
        cos_theta = torch.sqrt(1 - r_sq)
        phi = 2 * math.pi * torch.rand(num_rays, 1, device=device)
        dirs = sin_theta * torch.cos(phi) * u_vec \
               + sin_theta * torch.sin(phi) * v_vec + cos_theta * main_dir
    elif source_type == "mesh_source" and mesh is not None:
        mesh_verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        mesh_faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        tris = mesh_verts[mesh_faces]
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        cross_prod = torch.cross(e1, e2, dim=-1)
        areas = 0.5 * torch.norm(cross_prod, dim=-1)
        normals = cross_prod / (2 * areas.unsqueeze(-1) + 1e-12)
        cdf = torch.cumsum(areas, dim=0)
        cdf = cdf / cdf[-1]
        rand_cdf = torch.rand(num_rays, device=device)
        face_indices = torch.clamp(torch.searchsorted(cdf, rand_cdf), 0, mesh_faces.shape[0] - 1)
        r1 = torch.rand(num_rays, 1, device=device)
        r2 = torch.rand(num_rays, 1, device=device)
        sqrt_r1 = torch.sqrt(r1)
        u_bc = 1 - sqrt_r1
        v_bc = r2 * sqrt_r1
        w_bc = 1 - u_bc - v_bc
        selected_tris = tris[face_indices]
        origins = u_bc * selected_tris[:, 0] + v_bc * selected_tris[:, 1] + w_bc * selected_tris[:, 2]
        mesh_normals = normals[face_indices]
        r_sq = torch.rand(num_rays, 1, device=device)
        sin_theta = torch.sqrt(r_sq)
        cos_theta = torch.sqrt(1 - r_sq)
        phi = 2 * math.pi * torch.rand(num_rays, 1, device=device)
        ref = torch.zeros_like(mesh_normals)
        ref[:, 2] = 1.0
        ref[torch.abs(mesh_normals[:, 2]) > 0.999] = torch.tensor([0.0, 1.0, 0.0], device=device)
        local_u = torch.cross(ref, mesh_normals, dim=-1)
        local_u = local_u / torch.norm(local_u, dim=-1, keepdim=True)
        local_v = torch.cross(mesh_normals, local_u, dim=-1)
        dirs = sin_theta * torch.cos(phi) * local_u \
               + sin_theta * torch.sin(phi) * local_v + cos_theta * mesh_normals
    else:
        raise ValueError(f"Invalid source type: {source_type}")

    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    wavelengths = torch.full((num_rays, 1), float(wavelength_nm),
                             dtype=torch.float32, device=device)

    stokes = torch.zeros((num_rays, 4), dtype=torch.float32, device=device)
    stokes[:, 0] = 1.0
    if polarization == "s":
        stokes[:, 1] = 1.0
    elif polarization == "p":
        stokes[:, 1] = -1.0
    elif polarization == "random":
        pass
    else:
        ang = float(polarization) if isinstance(polarization, (int, float)) else 0.0
        stokes[:, 1] = math.cos(2 * math.radians(ang))
        stokes[:, 2] = math.sin(2 * math.radians(ang))

    s_ref = _choose_perpendicular(dirs)
    p_ref = torch.cross(dirs, s_ref, dim=-1)

    return {
        "origins": origins,
        "dirs": dirs,
        "wavelengths": wavelengths,
        "stokes": stokes,
        "s_ref": s_ref,
        "p_ref": p_ref,
        "ray_ids": torch.arange(num_rays, dtype=torch.long, device=device),
    }