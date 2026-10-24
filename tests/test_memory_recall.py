"""V2.3 加权召回与角色标注（契约 §2.6）。"""

from datetime import UTC, datetime, timedelta

from langchain_core.embeddings import Embeddings
from langgraph.store.memory import InMemoryStore

from app.memory.retriever import retrieve, retrieve_weighted
from app.memory.store import MemoryStore

_DIMS = 4
NOW = datetime(2026, 10, 21, tzinfo=UTC)


def _item(key: str, text: str, *, cos: float, days_old: float | None = 0.0, importance: float = 0.5,
          role: str | None = None, updated_at: str = "2026-10-21T00:00:00+00:00") -> dict:
    created = None
    if days_old is not None:
        created = (NOW - timedelta(days=days_old)).isoformat()
    return {
        "key": key,
        "text": text,
        "role": role,
        "kind": "memory",
        "score": cos,
        "importance": importance,
        "created_at": created,
        "updated_at": updated_at,
    }


class FakeStore:
    """只实现 retrieve_weighted 依赖的两个方法，便于精确控制 cos / 时间。"""

    def __init__(self, items: list[dict], facts: list[dict] | None = None) -> None:
        self.items = items
        self.facts = facts or []

    async def search_scored(self, agent_id, *, query, limit=5, overscan=50):
        return self.items[:overscan]

    async def current_facts(self, agent_id, *, limit=20):
        return self.facts[:limit]

    async def search(self, agent_id, query=None, limit=5):
        return [i["text"] for i in self.items[:limit]]


async def test_加权公式排序():
    # A: 0.6*1 + 0.25*1 + 0.15*1 = 1.0
    # C: 0.6*0.5 + 0.25*e^-1 + 0.15*0.5 ≈ 0.467
    # B: 0.6*0 + 0.25*1 + 0.15*0 = 0.25
    store = FakeStore(
        [
            _item("a", "A", cos=1.0, importance=1.0),
            _item("b", "B", cos=0.0, importance=0.0),
            _item("c", "C", cos=0.5, days_old=14.0, importance=0.5),
        ]
    )
    result = await retrieve_weighted(store, 1, query="q", now=NOW)
    assert result == ["A", "C", "B"]


async def test_缺created_at时recency取中性值():
    store = FakeStore(
        [
            _item("old", "旧数据", cos=0.0, days_old=None, importance=0.0),
            _item("ancient", "极旧", cos=0.0, days_old=3650.0, importance=0.0),
        ]
    )
    result = await retrieve_weighted(store, 1, query="q", now=NOW)
    assert result == ["旧数据", "极旧"]  # 0.125 > ≈0


async def test_确定性排序_score同则updated_at降序再key升序():
    store = FakeStore(
        [
            _item("b", "B", cos=0.5, updated_at="2026-10-21T00:00:00+00:00"),
            _item("a", "A", cos=0.5, updated_at="2026-10-21T00:00:00+00:00"),
            _item("c", "C", cos=0.5, updated_at="2026-10-20T00:00:00+00:00"),
        ]
    )
    assert await retrieve_weighted(store, 1, query="q", now=NOW) == ["A", "B", "C"]


async def test_纯余弦开关():
    # memory_weighted_recall=false 时调用方传 w_cosine=1、其余 0
    store = FakeStore(
        [
            _item("fresh", "新但余弦低", cos=0.1, importance=1.0),
            _item("stale", "旧但余弦高", cos=0.9, days_old=3650.0, importance=0.0),
        ]
    )
    weighted = await retrieve_weighted(store, 1, query="q", now=NOW)
    plain = await retrieve_weighted(
        store, 1, query="q", now=NOW, w_cosine=1.0, w_recency=0.0, w_importance=0.0
    )
    assert weighted[0] == "旧但余弦高"
    assert plain == ["旧但余弦高", "新但余弦低"]


async def test_角色标注前缀():
    store = FakeStore(
        [
            _item("u", "我叫阿飞", cos=0.9, role="user"),
            _item("a", "我给你倒茶", cos=0.8, role="assistant"),
            _item("n", "旧数据没有角色", cos=0.7, role=None),
        ]
    )
    result = await retrieve_weighted(store, 1, query="q", now=NOW)
    assert result == ["用户说过：我叫阿飞", "你说过：我给你倒茶", "旧数据没有角色"]


async def test_query为None退化为facts加记忆():
    store = FakeStore(
        [{"key": "k", "text": "普通记忆", "role": "user"}],
        facts=[{"text": "用户喜欢的茶=普洱"}, {"text": "普通记忆"}],
    )
    assert await retrieve_weighted(store, 1, query=None, limit=5) == ["用户喜欢的茶=普洱", "普通记忆"]


async def test_limit生效():
    store = FakeStore([_item(str(i), f"m{i}", cos=1.0 - i * 0.01) for i in range(10)])
    assert len(await retrieve_weighted(store, 1, query="q", limit=3, now=NOW)) == 3


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1] * _DIMS for _ in texts]

    def embed_query(self, text):
        return [0.1] * _DIMS


async def test_search原始返回不含角色前缀():
    store = MemoryStore(
        InMemoryStore(index={"dims": _DIMS, "embed": FakeEmbeddings(), "fields": ["text"]})
    )
    await store.put(1, "我叫阿飞", role="user")
    await store.put(1, "我给你倒茶", role="assistant")

    raw = await store.search(1, query="我叫阿飞", limit=5)
    assert set(raw) == {"我叫阿飞", "我给你倒茶"}
    assert all("用户说过" not in t and "你说过" not in t for t in raw)

    labeled = await retrieve_weighted(store, 1, query="我叫阿飞", now=NOW)
    assert "用户说过：我叫阿飞" in labeled
    assert "你说过：我给你倒茶" in labeled


async def test_retrieve保持旧语义返回原文():
    store = MemoryStore(
        InMemoryStore(index={"dims": _DIMS, "embed": FakeEmbeddings(), "fields": ["text"]})
    )
    await store.put(1, "我叫阿飞", role="user")
    assert await retrieve(store, 1, "我叫阿飞", limit=5) == ["我叫阿飞"]
