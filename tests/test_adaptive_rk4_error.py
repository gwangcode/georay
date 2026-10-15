"""
tests/test_adaptive_rk4_error.py
=================================
Error control validation for adaptive RK4.

Physics:
    The core guarantee of adaptive step-size algorithms is that the
    local error per step is <= tol.
    - rk_error should record the actual error estimate per step
    - After a rejected step, step size should halve
    - After an accepted step, step size should increase
"""
import torch
import trimesh
import pytest

from georay import AdvancedRayTracerPBRTGPU, generate_rays


def make_grin_box(device, extents=(2.0, 2.0, 4.0)):
    box = trimesh.creation.box(extents=list(extents))
    verts = torch.tensor(box.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(box.faces, dtype=torch.long, device=device)
    fm = torch.zeros((faces.shape[0], 2), dtype=torch.long, device=device)
    fm[:, 0] = 0
    fm[:, 1] = 1
    return verts, faces, fm


def test_rk4_error_below_tolerance():
    """
    检查所有接受步的 rk_error 应低于 rk_tol。

    注意：tracer 输出的是每步的 rk_err，无论接受或拒绝。
    拒绝步可能超过 tol，但 tracer 会重新减小步长。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device)

    def grin_fn(x, y, z, wl):
        return 1.5 - 0.1 * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": grin_fn},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.3, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tol = 1e-4
    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=200, ds_init=0.02,
                        ds_min=1e-4, ds_max=0.1,
                        rk_tol=tol, fresnel=False)

    rk_err = out["rk_error"][0]

    # Error on effective steps (active and position changed)
    pos = out["positions"][0]
    moved = (pos[1:] - pos[:-1]).norm(dim=-1) > 1e-8

    err_effective = rk_err[:len(moved)][moved]

    print("\n[Test 1] RK4 error below tolerance")
    print("  rk_tol = {:.2e}".format(tol))
    print("  Effective steps: {}".format(err_effective.shape[0]))
    print("  Max rk_error on effective steps: {:.3e}".format(
        err_effective.max().item() if len(err_effective) > 0 else 0.0))

    # Most steps (accepted) should have error < tol
    fraction_below = (err_effective < tol * 1.1).float().mean().item()
    print("  Fraction with rk_err < 1.1·tol: {:.2%}".format(fraction_below))

    # At least 80% of steps should satisfy error control
    assert fraction_below > 0.8, \
        "Only {:.2%} of steps have error below tolerance".format(fraction_below)


def test_rk4_error_decreases_with_smaller_tol():
    """
    rk_tol 越小，全局误差应越小。

    物理依据：自适应算法的总误差 ~ tol。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device)

    def grin_fn(x, y, z, wl):
        return 1.5 - 0.2 * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": grin_fn},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.3, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    # 基准（极细步长，几乎精确解）
    ref_tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    ref = ref_tracer.trace(rays, max_steps=5000, ds_init=0.001,
                            ds_min=0.001, ds_max=0.001, fresnel=False)
    x_ref = ref["positions"][0, -1, 0].item()

    print("\n[Test 2] RK4 error decreases with smaller tol")
    print("  Reference x_end (ds=0.001): {:.6f}".format(x_ref))

    prev_err = None
    for tol in [1e-2, 1e-3, 1e-4]:
        tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
        out = tracer.trace(rays, max_steps=2000, ds_init=0.02,
                            ds_min=1e-5, ds_max=0.1,
                            rk_tol=tol, fresnel=False)
        x_end = out["positions"][0, -1, 0].item()
        err = abs(x_end - x_ref)
        print("  rk_tol = {:.0e}: x_end = {:.6f}, |err| = {:.3e}".format(
            tol, x_end, err))

        if prev_err is not None:
            # 更小的 tol 应给出更小的误差
            assert err <= prev_err * 1.5, \
                "Smaller tol gave larger error: {:.3e} > {:.3e}".format(
                    err, prev_err)
        prev_err = err


def test_rk4_ds_increases_after_accepted_step():
    """
    接受步后，步长应该增加（因为误差 < tol）。
    """
    device = torch.device("cpu")
    verts, faces, fm = make_grin_box(device)

    # 极缓的 GRIN 场，几乎不会拒绝
    def grin_fn(x, y, z, wl):
        return 1.5 - 0.001 * (x ** 2 + y ** 2)

    materials = {
        0: {"n": 1.0, "reflectivity": 0.0, "diffuse": 0.0, "absorptivity": 0.0},
        1: {"n": 1.5, "reflectivity": 0.0, "diffuse": 0.0,
            "absorptivity": 0.0, "grin_field": grin_fn},
    }

    rays = generate_rays(
        source_type="grid", num_rays=1,
        origin=(0.3, 0.0, -3.0), direction=(0.0, 0.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 0.0),
        polarization="s", device=device)

    tracer = AdvancedRayTracerPBRTGPU(verts, faces, fm, materials, device)
    out = tracer.trace(rays, max_steps=200, ds_init=0.01,
                        ds_min=1e-4, ds_max=0.1,
                        rk_tol=1e-4, fresnel=False)

    ds_hist = out["ds_history"][0]
    initial_ds = ds_hist[0].item()

    print("\n[Test 3] ds increases after accepted step")
    print("  Initial ds: {:.6f}".format(initial_ds))
    print("  ds range: [{:.6f}, {:.6f}]".format(
        ds_hist.min().item(), ds_hist.max().item()))
    print("  ds history (first 10): {}".format(
        [round(v, 6) for v in ds_hist[:10].tolist()]))

    # 对于极缓的场，ds 应增长（趋于 h_max）
    assert ds_hist.max().item() > initial_ds * 1.2, \
        "ds should grow for weak GRIN field"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
