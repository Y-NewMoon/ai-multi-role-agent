"""FastAPI 异常处理器：把业务异常映射为合适的 HTTP 状态码。

- NotFoundError  → 404
- UpstreamError  → 502（外部依赖失败）
- 其他 AppError  → 400
- 未捕获异常     → 500（不泄漏回溯，仅记日志）

V3 M2：只对 `/api/v1/memory/*` 使用新错误信封
`{ok:false, contract:"1", error:{code,message}}`；既有 API 仍保持 `{detail:...}`。
"""

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import AppError, NotFoundError, UpstreamError
from app.services.memory_api_service import MemoryApiError

logger = logging.getLogger(__name__)


def _is_memory_path(path: str) -> bool:
    return path == "/api/v1/memory" or path.startswith("/api/v1/memory/")


def _envelope(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "contract": "1",
            "error": {"code": code, "message": message},
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(MemoryApiError)
    async def _handle_memory_api_error(
        request: Request, exc: MemoryApiError
    ) -> JSONResponse:
        return _envelope(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        if not _is_memory_path(request.url.path):
            return JSONResponse(
                status_code=422, content={"detail": jsonable_encoder(exc.errors())}
            )
        first = (exc.errors() or [{}])[0]
        loc = ".".join(str(part) for part in first.get("loc", ()))
        msg = str(first.get("msg") or "请求字段校验失败")
        message = f"{loc}: {msg}" if loc else msg
        return _envelope(422, "invalid_request", message)

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
        if _is_memory_path(request.url.path):
            return _envelope(500, "internal_error", "internal server error")
        return JSONResponse(status_code=500, content={"detail": "internal server error"})
