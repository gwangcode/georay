"""
GeoRay 自定义异常类。

设计原则：
    * 所有异常继承 GeoRayError，方便统一捕获
    * 关键异常同时继承 ValueError / RuntimeError，保持向后兼容
    * 每条错误信息包含"如何修复"的提示
"""


class GeoRayError(Exception):
    """所有 GeoRay 异常的基类。"""
    pass


class ConfigurationError(GeoRayError, ValueError):
    """配置错误（材质、场景、参数等）。

    同时继承 ValueError，保持与旧代码 `except ValueError` 的兼容性。
    """
    pass


class InvalidMaterialError(ConfigurationError):
    """材质配置无效（n <= 0、reflectivity 越界等）。"""
    pass


class InvalidRayError(GeoRayError, ValueError):
    """光线数据无效（缺少字段、shape 不匹配等）。"""
    pass


class GeometryError(GeoRayError, ValueError):
    """几何数据无效（空网格、退化三角形等）。"""
    pass


class BackendError(GeoRayError, RuntimeError):
    """后端错误（Warp 未安装、CUDA 不可用等）。"""
    pass