"""GeoRay: A GPU-Accelerated Ray Tracer for GRIN and Polarization Optics.

Public API:
    AdvancedRayTracerPBRTGPU  - Main ray tracer class
    BaseGRINField             - Base class for GRIN refractive index fields
    parse_grin_input_to_field - Parse GRIN config (constant/callable/dict)
    generate_rays             - Generate rays from various source types
    save_trace_output_to_npz  - Save trace results to compressed NPZ
    export_ray_history_to_stl - Export ray tracks to STL
"""

from .tracer import AdvancedRayTracerPBRTGPU
from .grin import BaseGRINField, parse_grin_input_to_field
from .sources import generate_rays
from .io import save_trace_output_to_npz, export_ray_history_to_stl

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "AdvancedRayTracerPBRTGPU",
    "BaseGRINField",
    "parse_grin_input_to_field",
    "generate_rays",
    "save_trace_output_to_npz",
    "export_ray_history_to_stl",
]