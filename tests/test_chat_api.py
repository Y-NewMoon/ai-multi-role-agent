"""对话 API 测试：SSE 端点帧格式与前置校验（不依赖真库 / 真 LLM）。"""

import httpx
from fastapi import FastAPI

from app.api.v1 import api_router
from app.core.error_handlers import register_exception_handlers
from app.core.errors import NotFoundError, UpstreamError


class _FakeService:
    """替身 ChatService：可注入“未找到”“上游异常”“正常事件序列”。"""

    def __init__(self, *, not_found: bool = False, events=None, exc: Exception | None = None) -> None:
        self._not_found = not_found
        self._events = events if events is not None else [
            ("token", {"text": "你"}),
            ("token", {"text": "好"}),
            ("done", {"session_id": 1, "agent_id": 1, "reply": "你好"}),
        ]
        self._exc = exc

    async def load_context(self, *, session_id: int):
        if self._not_found:
            raise NotFoundError(f"session {session_id} 不存在")
        return object()

    async def stream(self, *, context, message: str):
        if self._exc is not None:
            raise self._exc
        for event, data in self._events:
            yield event, data

    async def idle_stream(self, *, context):
        if self._exc is not None:
            raise self._exc
        for event, data in self._events:
            yield event, data

    async def clear_idle(self, *, session_id: int) -> list[int]:
        if self._not_found:
            raise NotFoundError(f"session {session_id} 不存在")
        return [1, 2]


def _app(service: _FakeService) -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    register_exception_handlers(app)
    app.state.chat_service = service
    return app


async def _post(app: FastAPI, url: str, payload: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(url, json=payload)


async def test_chat_stream_returns_sse_frames_in_order():
    resp = await _post(
        _app(_FakeService()), "/api/v1/chat/stream", {"session_id": 1, "message": "在吗"}
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert body.count("event: token") == 2
    assert body.index("event: token") < body.index("event: done")  # done 在 token 之后
    assert '"text": "你"' in body
    assert '"reply": "你好"' in body


async def test_chat_stream_returns_404_when_session_missing():
    resp = await _post(
        _app(_FakeService(not_found=True)),
        "/api/v1/chat/stream",
        {"session_id": 404, "message": "hi"},
    )

    assert resp.status_code == 404
    assert "不存在" in resp.text


async def test_chat_stream_emits_error_event_on_upstream_error():
    resp = await _post(
        _app(_FakeService(exc=UpstreamError("模型服务调用失败"))),
        "/api/v1/chat/stream",
        {"session_id": 1, "message": "hi"},
    )

    # 流已开始，HTTP 状态固定 200，异常以 error 事件表达
    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert "模型服务调用失败" in resp.text


async def test_chat_idle_stream_returns_sse_frames():
    resp = await _post(_app(_FakeService()), "/api/v1/chat/idle/stream", {"session_id": 1})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.count("event: token") == 2
    assert resp.text.index("event: token") < resp.text.index("event: done")


async def test_chat_idle_stream_returns_404_when_session_missing():
    resp = await _post(
        _app(_FakeService(not_found=True)), "/api/v1/chat/idle/stream", {"session_id": 404}
    )

    assert resp.status_code == 404
    assert "不存在" in resp.text


async def test_session_idle_clear_returns_removed_ids():
    resp = await _post(_app(_FakeService()), "/api/v1/sessions/1/idle/clear", {})

    assert resp.status_code == 200
    assert resp.json() == {"removed_message_ids": [1, 2]}


async def test_chat_stream_done_frame_carries_thinking():
    # M6：done 载荷的每条回复带上思考链，随 SSE 帧透出给前端
    events = [
        ("token", {"text": "你好"}),
        (
            "done",
            {
                "session_id": 1,
                "agent_id": 1,
                "reply": "你好",
                "replies": [
                    {"content": "你好", "message_id": 1, "thinking": "心里有点紧张"}
                ],
            },
        ),
    ]
    resp = await _post(
        _app(_FakeService(events=events)),
        "/api/v1/chat/stream",
        {"session_id": 1, "message": "在吗"},
    )

    assert resp.status_code == 200
    assert '"thinking": "心里有点紧张"' in resp.text
