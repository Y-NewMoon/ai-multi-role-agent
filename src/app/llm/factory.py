"""按 LLM_PROVIDER / EMBEDDING_PROVIDER 构造 Provider。"""

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.embedding import LocalBgeEmbeddings
from app.llm.fake import FakeEmbeddings, FakeLLMProvider
from app.llm.openai_compat import OpenAICompatProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "fake":
        # V2.4 §2.10 / C4：压测专用，不访问网络；生产不得使用
        return FakeLLMProvider(
            delay_ms=settings.fake_llm_delay_ms,
            reply=settings.fake_llm_reply,
        )
    if settings.llm_provider in ("deepseek", "openai", "openai_compat"):
        return OpenAICompatProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    raise ValueError(f"未知的 LLM_PROVIDER: {settings.llm_provider!r}")


def create_embedding_provider(settings: Settings) -> Embeddings:
    """构造向量嵌入器，供 AsyncPostgresStore 的 index.embed 使用。"""
    if settings.embedding_provider == "local":
        return LocalBgeEmbeddings(settings.embedding_model)
    if settings.embedding_provider == "fake":
        # V2.4 R1：仅压测 server 使用（确定性哈希向量），生产不得配置
        return FakeEmbeddings(settings.embedding_dims)
    raise ValueError(f"未知的 EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
