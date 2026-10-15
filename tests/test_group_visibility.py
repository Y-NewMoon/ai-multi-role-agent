"""群聊记忆可见性测试（方案 §8 第 5/6 条 · ADR-0004）。

必须跑真库 `roleplay_test`（真实 schema + pgvector）。每条用例造自己的角色/群，跑完清理。

- 第 5 条：某角色的私聊（private）记忆不得进入其它成员的 prompt / namespace。
- 第 6 条：群内共享信息（用户发言、角色公屏回复）对所有成员可读，
  且写入只落在各自 namespace（不建公共 namespace）。
"""

from sqlalchemy import delete, select

from app.agents.graph import build_graph
from app.db.models import Agent, GroupMessage, GroupSession
from app.db.models import GroupMember as GroupMemberRow
from app.memory.namespace import memory_namespace
from app.memory.store import MemoryStore
from app.prompts.roleplay_system import THINK_DELIMITER as T
from app.services.agent_service import AgentService
from app.services.group_service import GroupService

# 角色档案（唯一文本，便于在 prompt 里定位"这是谁的调用"）
_ALPHA = {"name": "阿尔法", "persona": "阿尔法的人设", "background": "阿尔法的背景"}
_BETA = {"name": "贝塔", "persona": "贝塔的人设", "background": "贝塔的背景"}


class _FakeLLM:
    """决策恒返回不可解析内容 → supervisor 走轮询兜底（确定性）；生成返回固定回复。"""

    def __init__(self, reply: str = "收到") -> None:
        self.reply = reply
        self.chat_calls: list[list[dict]] = []  # generate 节点的 prompt（astream）

    async def generate(self, messages: list[dict]) -> str:
        return ""  # 触发轮询兜底

    async def astream(self, messages: list[dict]):
        self.chat_calls.append(messages)
        yield self.reply


def _prompts_for(llm: _FakeLLM, persona_marker: str) -> list[list[dict]]:
    """按人设文本筛出某个成员收到的 prompt（system 内嵌 persona）。"""
    return [c for c in llm.chat_calls if any(persona_marker in m["content"] for m in c)]


async def _make_agents(session_factory, specs: list[dict]) -> list[int]:
    async with session_factory() as db:
        agents = [
            Agent(name=s["name"], persona_prompt=s["persona"], background=s["background"])
            for s in specs
        ]
        db.add_all(agents)
        await db.flush()
        ids = [a.id for a in agents]
        await db.commit()
        return ids


async def _make_group(session_factory, *, name: str, agent_ids: list[int]) -> int:
    async with session_factory() as db:
        group = GroupSession(name=name)
        db.add(group)
        await db.flush()
        for agent_id in agent_ids:
            db.add(GroupMemberRow(group_session_id=group.id, agent_id=agent_id))
        await db.commit()
        return group.id


async def _cleanup(session_factory, checkpointer, store, *, gid, agent_ids) -> None:
    async with session_factory() as db:
        await db.execute(delete(GroupMessage).where(GroupMessage.group_session_id == gid))
        await db.execute(delete(GroupMemberRow).where(GroupMemberRow.group_session_id == gid))
        await db.execute(delete(GroupSession).where(GroupSession.id == gid))
        await db.execute(delete(Agent).where(Agent.id.in_(agent_ids)))
        await db.commit()
    for agent_id in agent_ids:
        await checkpointer.adelete_thread(f"g{gid}:{agent_id}")
        ns = memory_namespace(agent_id)
        for item in await store.asearch(ns, limit=1000):
            await store.adelete(item.namespace, item.key)


def _service(real_session_factory, llm, ms, checkpointer, store) -> GroupService:
    return GroupService(
        session_factory=real_session_factory,
        memory_store=ms,
        llm=llm,
        graph_factory=lambda: build_graph(
            llm=llm, memory_store=ms, checkpointer=checkpointer, store=store
        ),
        checkpointer=checkpointer,
    )


async def test_group_private_memory_not_leaked_into_other_member(
    real_store, real_checkpointer, real_session_factory
):
    """第 5 条：A 的私聊记忆不出现在 B 的 prompt，也不落进 B 的 namespace。"""
    llm = _FakeLLM("收到")
    ms = MemoryStore(real_store)
    a_id, b_id = await _make_agents(real_session_factory, [_ALPHA, _BETA])
    gid = await _make_group(real_session_factory, name="可见性-私密", agent_ids=[a_id, b_id])

    try:
        # A 的私聊秘密：只在 A 自己的 namespace
        await ms.put(a_id, "A 的私聊秘密：保险箱口令 998")

        service = _service(real_session_factory, llm, ms, real_checkpointer, real_store)
        context = await service.load_context(gid)
        _ = [e async for e in service.stream(context=context, message="大家晚上好")]

        beta_prompts = _prompts_for(llm, _BETA["persona"])
        assert beta_prompts, "B 本轮应当收到过 prompt"
        assert all(
            "保险箱口令 998" not in m["content"] for p in beta_prompts for m in p
        ), "A 的私聊记忆不得进入 B 的 prompt"
        # B 的 namespace 里也不该有 A 的秘密
        assert all("保险箱口令" not in h for h in await ms.search(b_id, query="保险箱口令"))
        # 但 A 自己的 namespace 里仍在（私聊记忆属角色，未被清零）
        assert any("保险箱口令 998" in h for h in await ms.search(a_id, query="口令"))
    finally:
        await _cleanup(
            real_session_factory, real_checkpointer, real_store, gid=gid, agent_ids=[a_id, b_id]
        )


async def test_delete_agent_in_group_cascades(
    real_store, real_checkpointer, real_session_factory
):
    """删除已入群角色：外键级联生效（自动退群 + 其群消息移除），不抛外键错误。"""
    llm = _FakeLLM("你好")
    ms = MemoryStore(real_store)
    a_id, b_id = await _make_agents(real_session_factory, [_ALPHA, _BETA])
    gid = await _make_group(real_session_factory, name="可见性-删角色", agent_ids=[a_id, b_id])

    try:
        # 先跑一轮，让群里存在 A 的发言（agent_id=A 的 group_messages）
        service = _service(real_session_factory, llm, ms, real_checkpointer, real_store)
        context = await service.load_context(gid)
        _ = [e async for e in service.stream(context=context, message="打个招呼")]

        agents_service = AgentService(
            session_factory=real_session_factory,
            memory_store=ms,
            checkpointer=real_checkpointer,
        )
        await agents_service.delete(a_id)  # 修复前：group_members/group_messages 外键 RESTRICT → 报错

        async with real_session_factory() as db:
            members = set(
                (
                    await db.execute(
                        select(GroupMemberRow.agent_id).where(
                            GroupMemberRow.group_session_id == gid
                        )
                    )
                )
                .scalars()
                .all()
            )
            msgs = list(
                (
                    await db.execute(
                        select(GroupMessage)
                        .where(GroupMessage.group_session_id == gid)
                        .order_by(GroupMessage.id)
                    )
                )
                .scalars()
                .all()
            )
        assert a_id not in members and b_id in members, "删角色应自动退出其所在群"
        assert all(m.agent_id != a_id for m in msgs), "该角色的群消息应被级联清除"
        assert any(m.agent_id == b_id for m in msgs), "其它成员的消息应保留"
    finally:
        await _cleanup(
            real_session_factory, real_checkpointer, real_store, gid=gid, agent_ids=[b_id]
        )


async def test_group_shared_info_readable_by_all_members(
    real_store, real_checkpointer, real_session_factory
):
    """第 6 条：用户发言 + 角色公屏回复对所有成员可读；写入落在各自 namespace。"""
    llm = _FakeLLM("我也想吃火锅")
    ms = MemoryStore(real_store)
    a_id, b_id = await _make_agents(real_session_factory, [_ALPHA, _BETA])
    gid = await _make_group(real_session_factory, name="可见性-共享", agent_ids=[a_id, b_id])

    try:
        service = _service(real_session_factory, llm, ms, real_checkpointer, real_store)
        context = await service.load_context(gid)
        events = [e async for e in service.stream(context=context, message="今晚一起吃火锅")]

        # 用户发言已扇出到所有成员 namespace（不依赖向量命中，确定性可读）
        for agent_id in (a_id, b_id):
            assert any("今晚一起吃火锅" in h for h in await ms.search(agent_id, query="今晚吃什么"))
        # namespace 彼此不同（无公共 namespace）
        assert memory_namespace(a_id) != memory_namespace(b_id)

        # 快照落库：1 条用户 + 2 条角色（done 之前必须已落库）
        async with real_session_factory() as db:
            rows = list(
                (
                    await db.execute(
                        select(GroupMessage)
                        .where(GroupMessage.group_session_id == gid)
                        .order_by(GroupMessage.id)
                    )
                )
                .scalars()
                .all()
            )
        assert [r.role for r in rows] == ["user", "assistant", "assistant"]
        assert rows[0].agent_id is None  # 用户消息无 agent_id

        # 事件序列：done 在最后，且含发言顺序
        assert events[-1][0] == "done"
        assert events[-1][1]["order"] == [a_id, b_id]
        assert [e for e, _ in events].count("speaker") == 2
    finally:
        await _cleanup(
            real_session_factory, real_checkpointer, real_store, gid=gid, agent_ids=[a_id, b_id]
        )


async def test_group_writes_thinking_into_snapshot(
    real_store, real_checkpointer, real_session_factory
):
    """M6：群聊角色的内心独白随快照落库，且不混入公屏内容；用户行无思考。"""
    llm = _FakeLLM(f"我也想吃火锅{T}心里惦记着那家老店")
    ms = MemoryStore(real_store)
    a_id, b_id = await _make_agents(real_session_factory, [_ALPHA, _BETA])
    gid = await _make_group(real_session_factory, name="思考-群聊", agent_ids=[a_id, b_id])

    try:
        service = _service(real_session_factory, llm, ms, real_checkpointer, real_store)
        context = await service.load_context(gid)
        events = [e async for e in service.stream(context=context, message="今晚一起吃火锅")]

        done = events[-1][1]
        assert done["replies"], "应有角色回复"
        # done 载荷逐条带思考，且思考标记不混入公屏内容
        assert all(r["thinking"] == "心里惦记着那家老店" for r in done["replies"])
        assert all(T not in r["content"] for r in done["replies"])

        async with real_session_factory() as db:
            rows = list(
                (
                    await db.execute(
                        select(GroupMessage)
                        .where(GroupMessage.group_session_id == gid)
                        .order_by(GroupMessage.id)
                    )
                )
                .scalars()
                .all()
            )
        user_rows = [r for r in rows if r.role == "user"]
        assistant_rows = [r for r in rows if r.role == "assistant"]
        assert user_rows and user_rows[0].thinking is None  # 用户行无思考
        assert assistant_rows, "应有角色行"
        assert all(r.thinking == "心里惦记着那家老店" for r in assistant_rows)
    finally:
        await _cleanup(
            real_session_factory, real_checkpointer, real_store, gid=gid, agent_ids=[a_id, b_id]
        )


async def test_group_idle_writes_idle_messages_and_clear_only_idle(
    real_store, real_checkpointer, real_session_factory
):
    """自主闲聊：落库行全标记 is_idle；「清空闲聊」只删闲聊行/记忆，正常轮不受影响。"""
    llm = _FakeLLM("闲聊独有内容ABC")
    ms = MemoryStore(real_store)
    a_id, b_id = await _make_agents(real_session_factory, [_ALPHA, _BETA])
    gid = await _make_group(real_session_factory, name="闲聊-清空", agent_ids=[a_id, b_id])

    async def _rows():
        async with real_session_factory() as db:
            return list(
                (
                    await db.execute(
                        select(GroupMessage)
                        .where(GroupMessage.group_session_id == gid)
                        .order_by(GroupMessage.id)
                    )
                )
                .scalars()
                .all()
            )

    try:
        service = _service(real_session_factory, llm, ms, real_checkpointer, real_store)
        context = await service.load_context(gid)

        # 1) 一轮自主闲聊：每个成员各发一次，全部带 idle 标记，无用户行
        events = [e async for e in service.idle_stream(context=context)]
        assert [e for e, _ in events].count("speaker") == 2
        assert events[-1][0] == "done"
        done = events[-1][1]
        assert done["idle"] is True and done["message_id"] == 0
        assert len(done["replies"]) <= 10

        rows = await _rows()
        assert rows, "闲聊应落库"
        assert all(r.is_idle and r.role == "assistant" for r in rows)
        assert all(r.agent_id in (a_id, b_id) for r in rows)
        # 闲聊内容写入成员记忆
        assert any("闲聊独有内容ABC" in h for h in await ms.search(a_id, query="闲聊独有内容"))

        # 2) 再跑一轮正常对话：新行非 idle
        llm.reply = "正常回复XYZ"
        ctx2 = await service.load_context(gid)
        _ = [e async for e in service.stream(context=ctx2, message="今晚吃什么")]
        rows = await _rows()
        normal = [r for r in rows if not r.is_idle]
        assert any(r.role == "user" and r.content == "今晚吃什么" for r in normal)
        assert any(r.agent_id is not None and "正常回复XYZ" in r.content for r in normal)

        # 3) 清空闲聊：只删闲聊行，正常行保留
        removed = await service.clear_idle(group_session_id=gid)
        assert removed, "应清掉闲聊消息"
        left = await _rows()
        assert all(not r.is_idle for r in left)
        assert any(r.role == "user" and r.content == "今晚吃什么" for r in left)

        # 4) 闲聊记忆被清，正常记忆仍在
        assert all("闲聊独有内容ABC" not in h for h in await ms.search(a_id, query="闲聊独有内容"))
        assert any("正常回复XYZ" in h for h in await ms.search(a_id, query="正常回复"))
    finally:
        await _cleanup(
            real_session_factory, real_checkpointer, real_store, gid=gid, agent_ids=[a_id, b_id]
        )
