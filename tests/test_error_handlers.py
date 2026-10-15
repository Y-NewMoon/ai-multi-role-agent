"""异常处理器单测：业务异常 → HTTP 状态码映射。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.error_handlers import register_exception_handlers
from app.core.errors import AppError, NotFoundError, UpstreamError


def _client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/nf")
    async def _nf():
        raise NotFoundError("session 不存在")

    @app.get("/up")
    async def _up():
        raise UpstreamError("模型服务调用失败，请稍后再试")

    @app.get("/app")
    async def _app():
        raise AppError("业务错误")

    @app.get("/boom")
    async def _boom():
        raise ValueError("不该泄漏的细节")

    return TestClient(app, raise_server_exceptions=False)


def test_not_found_maps_to_404():
    r = _client().get("/nf")
    assert r.status_code == 404
    assert r.json()["detail"] == "session 不存在"


def test_upstream_maps_to_502():
    r = _client().get("/up")
    assert r.status_code == 502
    assert "模型服务调用失败" in r.json()["detail"]


def test_app_error_maps_to_400():
    r = _client().get("/app")
    assert r.status_code == 400


def test_unexpected_maps_to_500_without_leaking_detail():
    r = _client().get("/boom")
    assert r.status_code == 500
    assert r.json()["detail"] == "internal server error"
