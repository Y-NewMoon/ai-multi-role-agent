"""统一业务异常。"""


class AppError(Exception):
    """业务异常基类。"""


class NotFoundError(AppError):
    """资源不存在。"""


class UpstreamError(AppError):
    """外部依赖（LLM 等）调用失败。"""
