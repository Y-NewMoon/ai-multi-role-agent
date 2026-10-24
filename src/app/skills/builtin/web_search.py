"""`web_search`（契约 §2.5-3）：可注入 `SearchProvider`；无 key 返回 `unavailable`。

CI/单测用 `FakeSearchProvider`（不访问网络）；生产用 `HttpSearchProvider`。
"""

from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from app.core.errors import UpstreamError
from app.skills.base import BaseSkill, ToolContext, ToolResult


class SearchProvider(Protocol):
    """搜索后端协议：`available()` 为假时工具返回 `unavailable`。"""

    name: str

    def available(self) -> bool: ...

    async def search(self, query: str, top_k: int) -> list[dict]: ...


class FakeSearchProvider:
    """测试/CI 用：返回预置结果，不访问网络。"""

    name = "fake"

    def __init__(self, results: list[dict] | None = None, *, available: bool = True) -> None:
        self._results = results or []
        self._available = available

    def available(self) -> bool:
        return self._available

    async def search(self, query: str, top_k: int) -> list[dict]:
        return list(self._results)[:top_k]


class HttpSearchProvider:
    """HTTP 搜索后端：`WEB_SEARCH_URL` + `WEB_SEARCH_API_KEY`（契约 §2.5-3）。"""

    name = "http"

    def __init__(self, *, url: str = "", api_key: str = "", timeout_s: float = 5.0) -> None:
        self._url = url
        self._api_key = api_key
        self._timeout = timeout_s

    def available(self) -> bool:
        return bool(self._url and self._api_key)

    async def search(self, query: str, top_k: int) -> list[dict]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        params = {"q": query, "top_k": top_k}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._url, params=params, headers=headers)
            resp.raise_for_status()  # 5xx/429 → httpx.HTTPStatusError → 执行器按 upstream_error 处理
        data: Any = resp.json()
        items = data.get("results", []) if isinstance(data, dict) else []
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("snippet", "")),
            }
            for item in items
            if isinstance(item, dict)
        ]


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=3, ge=1, le=5)


def make_web_search_skill(provider: SearchProvider | None = None) -> BaseSkill:
    async def handler(ctx: ToolContext, args: WebSearchArgs) -> ToolResult:
        if provider is None or not provider.available():
            return ToolResult(
                ok=False,
                error_type="unavailable",
                message="搜索服务不可用（缺少 WEB_SEARCH_URL / WEB_SEARCH_API_KEY）",
            )
        try:
            results = await provider.search(args.query, args.top_k)
        except UpstreamError:
            raise
        except httpx.HTTPStatusError:
            raise
        if not results:
            return ToolResult(ok=False, error_type="empty", message="搜索无结果")
        return ToolResult(
            ok=True,
            data={"results": list(results)[: args.top_k], "provider": provider.name},
        )

    return BaseSkill(
        name="web_search",
        description="联网搜索给定查询并返回网页摘要",
        when_to_use="需要联网检索最新信息时使用",
        parameters=WebSearchArgs,
        handler=handler,
        timeout_s=5.0,
        # V2.4 B-01：无后端/无 key 时不下发给模型（直接调用仍返回 unavailable）
        available=lambda: provider is not None and provider.available(),
    )
