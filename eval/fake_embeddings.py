"""确定性向量（FakeEmbeddings）——评测专用，与 tests/conftest.py 的 `_FakeEmbeddings` 算法逐字一致。

算法（三处必须一致：conftest / eval / 新测试）：
sha256(text) → 按 dims 重复填充 → 每字节 / 255.0。

用途：
- `--ci` / 无 sentence-transformers 环境下给 LangGraph Store 的向量索引供向量；
- 单测里复现检索分数（算法不一致会导致分数对不上）。

注意：不要 import 测试模块，测试侧反过来 import 本模块。
"""

import hashlib

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """确定性向量（按文本哈希），不加载真模型，但保持 LangChain `Embeddings` 接口。"""

    def __init__(self, dims: int) -> None:
        self._dims = dims

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (digest * ((self._dims // len(digest)) + 1))[: self._dims]
        return [b / 255.0 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
