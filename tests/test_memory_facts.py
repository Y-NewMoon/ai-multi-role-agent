"""V2.3 事实与冲突（契约 §2.1 / §2.2 / §2.4）。"""

import hashlib

from langgraph.store.memory import InMemoryStore

import app.memory.store as store_module
from app.memory.extractor import ExtractedFact, Extraction
from app.memory.facts import render_facts, record_facts
from app.memory.store import FACT_PREFIX, MemoryStore, _memory_key


def _store() -> MemoryStore:
    return MemoryStore(InMemoryStore())


async def test_put_fact_幂等():
    store = _store()
    a = await store.put_fact(1, subject="用户", predicate="喜欢的茶", obj="普洱", importance=0.8)
    b = await store.put_fact(1, subject="用户", predicate="喜欢的茶", obj="普洱", importance=0.8)
    assert a.created is True and a.fact_key.startswith(FACT_PREFIX)
    assert b.created is False and b.fact_key == a.fact_key and b.superseded == []
    assert len(await store.current_facts(1)) == 1


async def test_overwrite_旧值失效且检索过滤():
    store = _store()
    await store.put_fact(1, subject="用户", predicate="喜欢的茶", obj="龙井", importance=0.6)
    result = await store.put_fact(1, subject="用户", predicate="喜欢的茶", obj="普洱", importance=0.8)

    assert len(result.superseded) == 1
    active = [f["object"] for f in await store.current_facts(1)]
    assert active == ["普洱"]
    # search 仍返回 list[str]，且不再返回已失效的 fact 文本
    assert await store.search(1) == []
    # 失效条目不删（审计可回溯）
    items = await store._store.asearch(("agent:1", "memories"), limit=100)  # noqa: SLF001
    superseded = [i.value for i in items if i.value.get("superseded_at")]
    assert len(superseded) == 1 and superseded[0]["object"] == "龙井"


async def test_keep_both_旧值不失效():
    store = _store()
    await store.put_fact(1, subject="用户", predicate="喜欢的茶", obj="龙井")
    result = await store.put_fact(
        1,
        subject="用户",
        predicate="喜欢的茶",
        obj="普洱",
        mode="keep_both_with_note",
    )
    assert result.superseded == []
    assert {f["object"] for f in await store.current_facts(1)} == {"龙井", "普洱"}
    cur = await store._store.aget(  # noqa: SLF001
        ("agent:1", "memories"), store_module._fact_cur_key(1, "用户", "喜欢的茶")
    )
    assert cur is not None and cur.value["fact_key"] == result.fact_key


async def test_失效后重新主张恢复active():
    store = _store()
    await store.put_fact(1, subject="用户", predicate="喜欢的茶", obj="龙井")
    await store.put_fact(1, subject="用户", predicate="喜欢的茶", obj="普洱")
    again = await store.put_fact(1, subject="用户", predicate="喜欢的茶", obj="龙井")
    assert again.created is False and again.superseded  # 普洱被取代
    assert {f["object"] for f in await store.current_facts(1)} == {"龙井"}


async def test_fact上限淘汰最弱(monkeypatch):
    monkeypatch.setattr(store_module, "_MAX_FACTS_PER_AGENT", 3)
    store = _store()
    for i, importance in enumerate([0.9, 0.1, 0.5, 0.2, 0.7]):
        await store.put_fact(1, subject="用户", predicate=f"p{i}", obj=f"o{i}", importance=importance)
    facts = await store.current_facts(1, limit=10)
    assert len(facts) == 3
    kept = {round(f["importance"], 1) for f in facts}
    assert kept == {0.9, 0.7, 0.5}  # 最弱的 0.1 / 0.2 被淘汰


async def test_fact_limit可配且超限先淘汰superseded():
    """R5/R9：上限可配；超限先删 superseded，active 永远优先保留。"""
    store = MemoryStore(InMemoryStore(), fact_limit=2)
    assert store._fact_limit == 2  # noqa: SLF001
    await store.put_fact(1, subject="用户", predicate="茶", obj="龙井", importance=0.9)
    await store.put_fact(1, subject="用户", predicate="茶", obj="普洱", importance=0.9)  # 龙井失效
    await store.put_fact(1, subject="用户", predicate="sort", obj="乌龙", importance=0.1)  # active 且最弱

    items = await store._store.asearch(("agent:1", "memories"), limit=100)  # noqa: SLF001
    facts = [i.value for i in items if i.value.get("kind") == "fact"]
    assert len(facts) == 2
    # superseded 的「龙井」先出局；active 的「普洱 / 乌龙」都保留（乌龙虽最弱）
    assert {f["object"] for f in facts} == {"普洱", "乌龙"}
    assert not any(f.get("superseded_at") for f in facts)


async def test_默认200下201次冲突后active非空():
    """R8 回归：同一 (用户, 茶) 连续改口 201 次，默认上限下不得把 active 饿死。"""
    store = _store()
    assert store._fact_limit == 200  # noqa: SLF001
    for i in range(201):
        await store.put_fact(1, subject="用户", predicate="茶", obj=f"第{i}种", importance=0.5)

    active = await store.current_facts(1, limit=500)
    assert len(active) == 1
    assert active[0]["object"] == "第200种"
    # fact_cur 与 active 一致，不指向已删除的 key
    cur = await store._store.aget(  # noqa: SLF001
        ("agent:1", "memories"), store_module._fact_cur_key(1, "用户", "茶")
    )
    assert cur is not None
    assert cur.value["fact_key"] == active[0]["key"]


async def test_旧key规则一字不改():
    store = _store()
    await store.put(1, "你好", role="user")
    expected = hashlib.sha1("1:你好".encode("utf-8")).hexdigest()[:32]
    assert _memory_key(1, "你好") == expected
    items = await store._store.asearch(("agent:1", "memories"), limit=10)  # noqa: SLF001
    assert [i.key for i in items] == [expected]


async def test_current_facts按importance降序():
    store = _store()
    await store.put_fact(1, subject="用户", predicate="a", obj="1", importance=0.3)
    await store.put_fact(1, subject="用户", predicate="b", obj="2", importance=0.9)
    await store.put_fact(1, subject="用户", predicate="c", obj="3", importance=0.6)
    assert [f["predicate"] for f in await store.current_facts(1)] == ["b", "c", "a"]


async def test_current_facts每agent隔离():
    store = _store()
    await store.put_fact(1, subject="用户", predicate="a", obj="1")
    await store.put_fact(2, subject="用户", predicate="a", obj="2")
    assert [f["object"] for f in await store.current_facts(1)] == ["1"]
    assert [f["object"] for f in await store.current_facts(2)] == ["2"]


async def test_record_facts单条失败不影响其它():
    class FlakyStore:
        def __init__(self) -> None:
            self.ok: list[str] = []

        async def put_fact(self, agent_id, *, subject, predicate, obj, **kwargs):
            if predicate == "坏的":
                raise RuntimeError("写失败")
            from app.memory.store import FactWrite

            self.ok.append(obj)
            return FactWrite("k", True, [])

    store = FlakyStore()
    extraction = Extraction(
        facts=[
            ExtractedFact("用户", "坏的", "x", 3),
            ExtractedFact("用户", "好的", "y", 4),
        ]
    )
    writes = await record_facts(store, 1, extraction, source_message_id=None, turn_id="s1:1")
    assert store.ok == ["y"] and len(writes) == 1


async def test_record_facts_importance除以5():
    class CaptureStore:
        def __init__(self) -> None:
            self.importance: list[float] = []

        async def put_fact(self, agent_id, *, subject, predicate, obj, importance, **kwargs):
            from app.memory.store import FactWrite

            self.importance.append(importance)
            return FactWrite("k", True, [])

    store = CaptureStore()
    await record_facts(
        store,
        1,
        Extraction(facts=[ExtractedFact("用户", "p", "o", 5)]),
        source_message_id=None,
        turn_id="s1:1",
    )
    assert store.importance == [1.0]


async def test_render_facts空列表返回空串():
    assert await render_facts([]) == ""


async def test_render_facts_limit与importance降序():
    facts = [
        {"subject": "用户", "predicate": f"p{i}", "object": f"o{i}", "importance": i / 10}
        for i in range(1, 6)
    ]
    text = await render_facts(facts, limit=3)
    assert text.startswith("【你记得的事】")
    lines = text.splitlines()[1:]
    assert lines == ["- 用户p5=o5", "- 用户p4=o4", "- 用户p3=o3"]


async def test_render_facts_keep_both只给最近旧值加注():
    facts = [
        {
            "subject": "用户",
            "predicate": "喜欢的茶",
            "object": "普洱",
            "importance": 0.9,
            "updated_at": "2026-10-21T02:00:00+00:00",
        },
        {
            "subject": "用户",
            "predicate": "喜欢的茶",
            "object": "龙井",
            "importance": 0.8,
            "updated_at": "2026-10-20T02:00:00+00:00",
        },
        {
            "subject": "用户",
            "predicate": "喜欢的茶",
            "object": "碧螺春",
            "importance": 0.7,
            "updated_at": "2026-10-19T02:00:00+00:00",
        },
    ]
    text = await render_facts(facts)
    assert "- 用户喜欢的茶=普洱" in text
    assert "- 用户喜欢的茶=龙井（你以前说龙井）" in text
    assert "碧螺春" not in text  # 更早的不注入
