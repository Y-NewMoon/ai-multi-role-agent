"""本地 embedding provider（方案 3.6 / 3.12）。

用 sentence-transformers 加载本地 bge-m3（1024 维），实现 LangChain `Embeddings`
接口，供 `AsyncPostgresStore` 的 `index.embed` 做向量索引。

性能/环境提示（本机实测 2026-10-11）：
- Windows 上 PyPI 的 torch 为 CPU 构建，只有约 118MB，不占用 8GB 显存；
  bge-m3 首次加载约 13s，单条编码 <1s。
- bge-m3 权重已缓存于本地 HF cache，因此**默认离线加载**，避开 hf-mirror
  在线校验/大文件下载导致的挂起（实测会卡在 1.3-1.6GB）。
- 首次下载权重时需显式覆盖环境变量：`HF_HUB_OFFLINE=0` 且
  `HF_ENDPOINT=https://hf-mirror.com`（并建议 `HF_HUB_DISABLE_XET=1`）。
"""

import os

# 必须在导入 huggingface_hub（sentence_transformers 的依赖）之前设置，
# 因为 huggingface_hub.constants 在 import 时读取该变量。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from langchain_core.embeddings import Embeddings  # noqa: E402

# 短名 -> HF 仓库 id。EMBEDDING_MODEL 默认值保持 "bge-m3"（不改动 .env）。
_MODEL_ALIASES = {"bge-m3": "BAAI/bge-m3"}


class LocalBgeEmbeddings(Embeddings):
    """bge-m3 本地嵌入，输出维度固定 1024（方案 3.6 定死，建表后不可改）。"""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        repo_id = _MODEL_ALIASES.get(model_name, model_name)
        self._model = SentenceTransformer(repo_id)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vector]
