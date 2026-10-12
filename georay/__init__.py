"""GeoRay: A GPU-Accelerated Ray Tracer for GRIN and Polarization Optics.

Public API:
    AdvancedRayTracerPBRTGPU  - PyTorch ray tracer (BVH fallback)
    WarpTracer                - Full-Warp-kernel ray tracer (fastest)
    BaseGRINField             - Base class for GRIN refractive index fields
    parse_grin_input_to_field - Parse GRIN config
    generate_rays             - Generate rays from various source types
    save_trace_output_to_npz  - Save trace results to compressed NPZ
    export_ray_history_to_stl - Export ray tracks to STL
"""

from .tracer import AdvancedRayTracerPBRTGPU
from .grin import BaseGRINField, parse_grin_input_to_field
from .sources import generate_rays
from .io import save_trace_output_to_npz, export_ray_history_to_stl

# WarpTracer 是可选的（需要 warp-lang）
try:
    from .warp_tracer import WarpTracer
    _HAS_WARP = True
except ImportError:
    WarpTracer = None
    _HAS_WARP = False

__version__ = "1.1.0"
__author__ = "Gang Wang"

__all__ = [
    "AdvancedRayTracerPBRTGPU",
    "WarpTracer",
    "BaseGRINField",
    "parse_grin_input_to_field",
    "generate_rays",
    "save_trace_output_to_npz",
    "export_ray_history_to_stl",
]