"""GeoRay: A GPU-Accelerated Ray Tracer for GRIN and Polarization Optics.

Public API:
    AdvancedRayTracerPBRTGPU  - PyTorch-based ray tracer
    WarpTracer                - Full-Warp-kernel ray tracer (fastest)
    BaseGRINField             - Base class for GRIN refractive index fields
    parse_grin_input_to_field - Parse GRIN config
    generate_rays             - Generate rays from various source types
    save_trace_output_to_npz  - Save trace results to compressed NPZ
    export_ray_history_to_stl - Export ray tracks to STL

Exceptions:
    GeoRayError               - Base class
    ConfigurationError        - Invalid configuration
    InvalidMaterialError      - Invalid material config
    InvalidRayError           - Invalid ray data
    GeometryError             - Invalid geometry
    BackendError              - Backend (Warp, CUDA) errors
"""

from .tracer import AdvancedRayTracerPBRTGPU
from .grin import BaseGRINField, parse_grin_input_to_field
from .sources import generate_rays
from .io import save_trace_output_to_npz, export_ray_history_to_stl

from .exceptions import (
    GeoRayError,
    ConfigurationError,
    InvalidMaterialError,
    InvalidRayError,
    GeometryError,
    BackendError,
)

try:
    from .warp_tracer import WarpTracer
except ImportError:
    WarpTracer = None

__version__ = "1.2.0"
__author__ = "Gang Wang"

__all__ = [
    # Tracers
    "AdvancedRayTracerPBRTGPU",
    "WarpTracer",
    # GRIN
    "BaseGRINField",
    "parse_grin_input_to_field",
    # Rays
    "generate_rays",
    # I/O
    "save_trace_output_to_npz",
    "export_ray_history_to_stl",
    # Exceptions
    "GeoRayError",
    "ConfigurationError",
    "InvalidMaterialError",
    "InvalidRayError",
    "GeometryError",
    "BackendError",
]