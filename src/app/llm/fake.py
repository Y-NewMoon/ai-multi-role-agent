"""Fake LLM Provider（V2.4 §2.10 / C4）：**只**用于压测与测试，绝不访问网络。

生产默认仍是 `deepseek`；`LLM_PROVIDER=fake` 仅供本地压测/单测，README 有警告。
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator

from langchain_core.embeddings import Embeddings

from app.llm.base import ChatMessage, LLMProvider, LLMReply


class FakeLLMProvider(LLMProvider):
    """固定回复 + 固定延迟；`astream` 按小块逐段 yield，模拟流式首 token。"""

    def __init__(
        self,
        *,
        delay_ms: int = 10,
        reply: str = "（压测假回复）",
        chunk_chars: int = 4,
    ) -> None:
        self._delay_s = max(0, int(delay_ms)) / 1000.0
        self._reply = reply or "（压测假回复）"
        self._chunk_chars = max(1, int(chunk_chars))
        self._model = "fake"
        # 供测试断言“确实调了几次”（不计费，不写 trace）
        self.calls = 0

    async def generate(self, messages: list[ChatMessage]) -> str:
        self.calls += 1
        await asyncio.sleep(self._delay_s)
        return self._reply

    async def astream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.calls += 1
        text = self._reply
        for start in range(0, len(text), self._chunk_chars):
            await asyncio.sleep(self._delay_s)
            yield text[start : start + self._chunk_chars]

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
    ) -> LLMReply:
        self.calls += 1
        await asyncio.sleep(self._delay_s)
        return LLMReply(content=self._reply)


class FakeEmbeddings(Embeddings):
    """确定性哈希向量（`EMBEDDING_PROVIDER=fake`，压测/测试专用，不加载模型）。

    R1：20 并发下 bge-m3 的 CPU 争用会淹没 fake LLM 的首 token 指标；压测 server
    改用本嵌入器（dims 必须与建表时一致），只衡量服务链路本身。检索质量无意义，
    因此压测同时关闭世界书（`LORE_ENABLED=false`），不得用于生产/消融。
    """

    def __init__(self, dims: int = 1024) -> None:
        self._dims = max(1, int(dims))

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        raw = (digest * ((self._dims // len(digest)) + 1))[: self._dims]
        return [b / 255.0 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
