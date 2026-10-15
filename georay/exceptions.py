"""
GeoRay custom exception classes.

Design principles:
    * All exceptions inherit from GeoRayError for unified catching.
    * Critical exceptions also inherit ValueError / RuntimeError for
      backward compatibility.
    * Every error message includes a "how to fix" hint.
"""


class GeoRayError(Exception):
    """Base class for all GeoRay exceptions."""
    pass


class ConfigurationError(GeoRayError, ValueError):
    """Invalid configuration (materials, scene, parameters, etc.).

    Also inherits ValueError for backward compatibility with code that
    catches `except ValueError`.
    """
    pass


class InvalidMaterialError(ConfigurationError):
    """Invalid material configuration (n <= 0, reflectivity out of range, etc.)."""
    pass


class InvalidRayError(GeoRayError, ValueError):
    """Invalid ray data (missing fields, shape mismatch, etc.)."""
    pass


class GeometryError(GeoRayError, ValueError):
    """Invalid geometry (empty mesh, degenerate triangles, etc.)."""
    pass


class BackendError(GeoRayError, RuntimeError):
    """Backend error (Warp not installed, CUDA unavailable, etc.)."""
    pass
