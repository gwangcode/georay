# georay/intersectors.py
"""Intersector 工厂：自动选择 Warp / OptiX / 纯 PyTorch BVH。"""
import os
import torch


def make_intersector(vertices, faces, device, prefer="auto"):
    """
    创建网格求交器。

    Args:
        vertices, faces: 网格数据。
        device: torch.device。
        prefer: "auto" | "warp" | "optix" | "bvh"

    Returns:
        求交器实例。
    """
    env = os.environ.get("GEORAY_INTERSECTOR", "").lower()
    if env in ("warp", "optix", "bvh"):
        prefer = env

    if prefer == "bvh":
        from .bvh import BVHMeshIntersector
        return BVHMeshIntersector(vertices, faces, device)

    if prefer == "warp":
        from .warp_bvh import WarpMeshIntersector
        return WarpMeshIntersector(vertices, faces, device)

    if prefer == "optix":
        from .optix_bvh import OptixMeshIntersector
        return OptixMeshIntersector(vertices, faces, device)

    # auto
    if device.type == "cuda":
        try:
            import warp  # noqa
            from .warp_bvh import WarpMeshIntersector
            print("[GeoRay] Using Warp BVH.")
            return WarpMeshIntersector(vertices, faces, device)
        except Exception as e:
            print(f"[GeoRay] Warp unavailable ({e}); trying OptiX...")
            try:
                from .optix_bvh import OptixMeshIntersector
                print("[GeoRay] Using OptiX.")
                return OptixMeshIntersector(vertices, faces, device)
            except Exception as e2:
                print(f"[GeoRay] OptiX unavailable ({e2}); falling back to BVH.")
    from .bvh import BVHMeshIntersector
    print("[GeoRay] Using pure-PyTorch BVH.")
    return BVHMeshIntersector(vertices, faces, device)