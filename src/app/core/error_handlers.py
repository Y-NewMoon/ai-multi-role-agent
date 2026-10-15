"""FastAPI 异常处理器：把业务异常映射为合适的 HTTP 状态码。

- NotFoundError  → 404
- UpstreamError  → 502（外部依赖失败）
- 其他 AppError  → 400
- 未捕获异常     → 500（不泄漏回溯，仅记日志）
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import AppError, NotFoundError, UpstreamError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def _handle_not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(UpstreamError)
    async def _handle_upstream(request: Request, exc: UpstreamError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})
