"""V2.3 用户档案（契约 §2.5）。"""

import logging

from langchain_core.embeddings import Embeddings
from langgraph.store.memory import InMemoryStore

from app.memory.profile import PROFILE_HEADER, PROFILE_SLOTS, render_profile, update_profile
from app.memory.store import MemoryStore

_DIMS = 4


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * _DIMS for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * _DIMS


def _store(with_index: bool = False) -> MemoryStore:
    if with_index:
        return MemoryStore(
            InMemoryStore(index={"dims": _DIMS, "embed": FakeEmbeddings(), "fields": ["text"]})
        )
    return MemoryStore(InMemoryStore())


def test_槽位固定四个():
    assert PROFILE_SLOTS == ("称呼", "禁忌", "关系阶段", "重要日期")


async def test_update_profile写入白名单并返回完整档案():
    store = _store()
    result = await update_profile(store, 1, {"称呼": "阿林", "禁忌": "不喝浓茶"})
    assert result == {"称呼": "阿林", "禁忌": "不喝浓茶"}
    assert await store.get_profile(1) == result


async def test_update_profile同槽位覆盖():
    store = _store()
    await update_profile(store, 1, {"称呼": "小林"})
    assert await update_profile(store, 1, {"称呼": "阿林"}) == {"称呼": "阿林"}


async def test_update_profile空串清除槽位():
    store = _store()
    await update_profile(store, 1, {"称呼": "阿林", "禁忌": "不喝浓茶"})
    assert await update_profile(store, 1, {"禁忌": ""}) == {"称呼": "阿林"}


async def test_update_profile未知槽位忽略并告警(caplog):
    store = _store()
    with caplog.at_level(logging.WARNING):
        result = await update_profile(store, 1, {"口头禅": "好呀", "称呼": "阿林"})
    assert result == {"称呼": "阿林"}
    assert any("口头禅" in record.message for record in caplog.records)


async def test_render_profile空档案返回空串():
    assert await render_profile({}) == ""
    assert await render_profile({"称呼": "  "}) == ""


async def test_render_profile固定顺序():
    text = await render_profile({"重要日期": "生日 5 月 1 日", "称呼": "阿林"})
    assert text == f"{PROFILE_HEADER}\n- 称呼：阿林\n- 重要日期：生日 5 月 1 日"


async def test_profile不进向量召回():
    store = _store(with_index=True)
    await update_profile(store, 1, {"称呼": "阿林"})
    await store.put(1, "普通记忆", role="user")
    assert await store.search(1) == ["普通记忆"]
    candidates = await store.search_scored(1, query="阿林")
    assert [c["kind"] for c in candidates] == ["memory"]
    assert all("阿林" not in c["text"] for c in candidates)


async def test_profile每agent隔离():
    store = _store()
    await update_profile(store, 1, {"称呼": "阿林"})
    await update_profile(store, 2, {"称呼": "小周"})
    assert await store.get_profile(1) == {"称呼": "阿林"}
    assert await store.get_profile(2) == {"称呼": "小周"}
