"""内置工具单测 ②：read_lore / set_reminder（契约 §2.5-4~5，含真库幂等）。"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select

from app.db.models import Agent, Reminder, Session
from app.lore.retriever import LoreHit
from app.skills.base import ToolCall, ToolContext
from app.skills.builtin.read_lore import make_read_lore_skill
from app.skills.builtin.reminder import make_set_reminder_skill, parse_when
from app.skills.executor import ToolExecutor
from app.skills.registry import SkillRegistry

TZ = ZoneInfo("Asia/Shanghai")


async def _call(skill, args, ctx=None, **kw):
    ex = ToolExecutor(SkillRegistry([skill]), **kw)
    ctx = ctx or ToolContext(agent_id=1, session_id=2)
    return await ex.execute(ToolCall(id="c", name=skill.name, arguments=args), ctx)


class _FakeRetriever:
    """最小检索器替身：提供 `retrieve` + `_load_rows`（read_lore 只依赖这两个）。"""

    def __init__(self, *, reranked=None, injected=None, rows=None):
        self._reranked = reranked or []
        self._injected = injected or []
        self._rows = rows or {}

    async def retrieve(self, *, agent_id, query):  # noqa: ANN001
        return SimpleNamespace(reranked=self._reranked, injected=self._injected)

    async def _load_rows(self, *, agent_id, refs):  # noqa: ANN001
        return {ref: self._rows[ref] for ref in refs if ref in self._rows}


# ---- read_lore ----

async def test_read_lore_unavailable_without_retriever():
    res = await _call(make_read_lore_skill(None), {"query": "茶馆"})
    assert res.error_type == "unavailable"


async def test_read_lore_empty_when_no_hits():
    res = await _call(make_read_lore_skill(_FakeRetriever()), {"query": "茶馆"})
    assert res.error_type == "empty"


async def test_read_lore_uses_injected_chunks():
    hit = LoreHit(
        source_ref="yunqi_tea:01",
        header_path="茶馆",
        content="云栖茶馆",
        token_count=4,
        score=0.9,
        routes=("vector",),
    )
    retriever = _FakeRetriever(reranked=[("yunqi_tea:01", 0.9)], injected=[hit])
    res = await _call(make_read_lore_skill(retriever), {"query": "茶馆", "top_k": 5})
    assert res.ok
    assert res.data["chunks"][0]["source_ref"] == "yunqi_tea:01"
    assert res.data["chunks"][0]["content"] == "云栖茶馆"


async def test_read_lore_loads_missing_rows():
    row = SimpleNamespace(source_ref="qingxi_town:03", header_path="水乡", content="青溪镇")
    retriever = _FakeRetriever(reranked=[("qingxi_town:03", 0.7)], rows={"qingxi_town:03": row})
    res = await _call(make_read_lore_skill(retriever), {"query": "青溪镇", "top_k": 5})
    assert res.ok
    assert res.data["chunks"][0]["content"] == "青溪镇"
    assert res.data["chunks"][0]["score"] == 0.7


async def test_read_lore_rejects_bad_top_k():
    retriever = _FakeRetriever()
    assert (await _call(make_read_lore_skill(retriever), {"query": "x", "top_k": 11})).error_type == "invalid_arguments"
    assert (await _call(make_read_lore_skill(retriever), {"query": ""})).error_type == "invalid_arguments"


# ---- set_reminder：时间解析 ----

def test_parse_when_iso_and_local():
    now = datetime.now(TZ)
    assert parse_when("2099-01-01T09:00:00+08:00", now=now) is not None
    local = parse_when("2099-01-01 09:00", now=now)
    assert local is not None and local.tzinfo is not None


def test_parse_when_past_and_bad():
    now = datetime.now(TZ)
    assert parse_when((now - timedelta(hours=1)).isoformat(), now=now) is None
    assert parse_when("not-a-date", now=now) is None
    assert parse_when("", now=now) is None


async def test_set_reminder_invalid_inputs():
    skill = make_set_reminder_skill(None)
    assert (await _call(skill, {"when": "not-a-date", "content": "开会"})).error_type == "invalid_time"
    assert (await _call(skill, {"when": "2099-01-01 09:00", "content": ""})).error_type == "invalid_content"
    assert (await _call(skill, {"when": "2099-01-01 09:00", "content": "x"})).error_type == "unavailable"


# ---- set_reminder：真库幂等（需 TEST_DATABASE_URL） ----

async def test_set_reminder_idempotent(real_session_factory):
    factory = real_session_factory
    async with factory() as db:
        agent = Agent(name="幂等角色", persona_prompt="p")
        db.add(agent)
        await db.flush()
        sess = Session(agent_id=agent.id, title="幂等会话")
        db.add(sess)
        await db.flush()
        agent_id, session_id = agent.id, sess.id
        await db.commit()

    try:
        skill = make_set_reminder_skill(factory)
        ctx = ToolContext(agent_id=agent_id, session_id=session_id)
        args = {"when": "2099-01-01 09:00", "content": "提醒我开会"}
        first = await _call(skill, args, ctx=ctx)
        second = await _call(skill, args, ctx=ctx)
        assert first.ok and second.ok
        assert first.data["status"] == "pending"
        assert first.data["idempotent"] is False
        assert second.data["idempotent"] is True
        assert first.data["reminder_id"] == second.data["reminder_id"]

        async with factory() as db:
            rows = (
                (await db.execute(select(Reminder).where(Reminder.session_id == session_id)))
                .scalars()
                .all()
            )
            assert len(rows) == 1  # 同参数两次只落一条
            assert rows[0].status == "pending"  # 未确认不激活
    finally:
        async with factory() as db:
            await db.execute(delete(Reminder).where(Reminder.session_id == session_id))
            await db.execute(delete(Session).where(Session.id == session_id))
            await db.execute(delete(Agent).where(Agent.id == agent_id))
            await db.commit()
