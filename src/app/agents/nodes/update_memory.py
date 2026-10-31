"""节点：写入记忆 + 触发滚动摘要（M2 / M5）+ V2.3 事实抽取与 L2 触发。

接缝 4（方案 3.11A#4）：以 (agent_id, content) 为单位写入，不以 session 为单位。
接缝 2：写入该 agent 自己的 namespace（经注入的 Store，图内不碰业务表）。
方案 3.10：工作记忆超过 trigger_rounds 轮时，把最旧 chunk_rounds 轮压缩为 1 条摘要
（落 key=summary:{session_id}），并用 RemoveMessage 裁剪 checkpointer 侧窗口；
`messages` 表仍保留全量用于展示（由服务层写，图内不写业务表）。

V2.3（§2.9）：
- 写完本轮记忆后（D2：回复之后），若 `memory_extract_enabled` 且非闲聊且用户文本
  ≥ `memory_extract_min_chars`，调 `extract` → `record_facts` + `update_profile`；
  任何异常只记 log（fail-open，D8）。
- 滚动摘要**成功**后 L1 合并计数 +1；达到 `memory_l2_every_merges` 时重合成 L2。
- 滚动摘要整条链路 fail-open：失败只记日志、缓冲不裁剪，下一轮重试。
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from app.agents.state import RolePlayState
from app.llm.base import LLMProvider
from app.memory.extractor import extract
from app.memory.facts import record_facts
from app.memory.l2 import maybe_update_l2
from app.memory.profile import update_profile
from app.memory.store import MemoryStore
from app.memory.summarizer import summarize

logger = logging.getLogger(__name__)


def _last_human(messages: list):
    """本轮用户消息（最后一条 HumanMessage）。"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return None


def _round_count(messages: list) -> int:
    """轮数 = 用户消息条数（每轮恰有一条用户消息）。"""
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _old_rounds(messages: list, chunk_rounds: int) -> list:
    """截取最旧 chunk_rounds 轮（含各轮全部角色回复），从第 chunk_rounds+1 条用户消息处断开。"""
    seen = 0
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            seen += 1
            if seen == chunk_rounds + 1:
                return messages[:i]
    return list(messages)


def _is_idle_message(m) -> bool:
    """是否为自主闲聊消息（generate 打的额外标记）。"""
    return bool((getattr(m, "additional_kwargs", None) or {}).get("idle"))


async def update_memory(
    state: RolePlayState,
    memory_store: MemoryStore,
    llm: LLMProvider,
    *,
    trigger_rounds: int = 10,
    chunk_rounds: int = 5,
    extract_enabled: bool = False,
    extract_min_chars: int = 8,
    extract_timeout_s: float = 8.0,
    fact_mode: str = "overwrite",
    profile_enabled: bool = True,
    l2_enabled: bool = True,
    l2_every_merges: int = 3,
    l2_max_chars: int = 300,
) -> dict:
    agent_id = state["agent_id"]
    session_id = state["session_id"]
    messages = list(state.get("messages", []))
    is_idle = bool(state.get("idle_scope"))

    # V2.4 §2.9：降级轮（LLM 超时兜底）不写任何记忆/摘要，也不触发事实抽取
    if state.get("degraded"):
        logger.warning(
            "降级轮跳过记忆写入（agent_id=%s, session_id=%s）", agent_id, session_id
        )
        return {}
    # 不变量 7：抽取总开关关闭时，V2.3 的 L2 计数/重合成也必须停（否则 Store 仍被写）
    l2_enabled = bool(l2_enabled and extract_enabled)
    profile_enabled = bool(profile_enabled and extract_enabled)

    # 1) 写入本轮新记忆（粒度 = agent_id + content，同内容覆盖）
    #    M5：一轮可含多条角色回复，逐条写入；无 replies 时回退为最后一条 AI（兼容旧路径）。
    #    自主闲聊轮没有真实用户发言，跳过 human 写入，避免把历史用户消息当本轮重复入库。
    human = _last_human(messages)
    user_text = str(human.content).strip() if human is not None else ""
    if not is_idle and user_text:
        await memory_store.put(agent_id, user_text, role="user")

    replies = [str(r).strip() for r in (state.get("replies") or [])]
    replies = [r for r in replies if r]
    if not replies:
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai is not None and str(last_ai.content).strip():
            replies = [str(last_ai.content).strip()]
    for content in replies:
        await memory_store.put(agent_id, content, role="assistant")

    # 2) V2.3：回复后抽取事实 + profile（失败 fail-open，不阻断回复）
    if (
        extract_enabled
        and not is_idle
        and human is not None
        and len(user_text) >= extract_min_chars
        and replies
    ):
        try:
            current_profile = await memory_store.get_profile(agent_id)
            extraction = await extract(
                llm,
                user_text,
                "\n".join(replies),
                timeout_s=extract_timeout_s,
                current_profile=current_profile,
            )
            if extraction is not None:
                # 契约 §2.1/§2.9：私聊 s{session_id}:{轮次}，群聊 g{group_id}:{轮次}
                session_tag = str(session_id)
                if not session_tag.startswith("g"):
                    session_tag = f"s{session_tag}"
                turn_id = f"{session_tag}:{_round_count(messages)}"
                await record_facts(
                    memory_store,
                    agent_id,
                    extraction,
                    source_message_id=getattr(human, "id", None),
                    turn_id=turn_id,
                    mode=fact_mode,
                )
                if profile_enabled and extraction.profile:
                    await update_profile(memory_store, agent_id, extraction.profile)
        except Exception as exc:  # noqa: BLE001 - 记忆是增强项，不能成为可用性风险（D8）
            logger.warning("事实抽取/写入失败（agent=%s）：%s", agent_id, exc)

    # 3) 超阈值则滚动摘要：压缩最旧 chunk_rounds 轮，并裁剪工作记忆窗口
    #    fail-open（D8）：摘要链路任何一步失败（模型抖动、写库失败）都只记日志，
    #    本轮记忆照常保留，缓冲不裁剪，下一轮再试——否则一次失败会让 /memory/turn
    #    永久 502（缓冲卡在阈值之上，每次重试都在同一处再炸一次）。
    if _round_count(messages) <= trigger_rounds:
        return {}

    try:
        old_chunk = _old_rounds(messages, chunk_rounds)
        # 阻止式：摘要输入剔除自主闲聊消息（闲聊永不被并入摘要），但裁剪窗口仍包含它们。
        summarize_input = [m for m in old_chunk if not _is_idle_message(m)]
        existing = await memory_store.get_summary(agent_id, session_id)
        summary_text = await summarize(llm, summarize_input, existing_summary=existing)
        if not summary_text:
            return {}

        await memory_store.put_summary(agent_id, session_id, summary_text)

        # 4) V2.3：L1 滚动合并成功 → L2 计数 +1，达到阈值则重合成（失败不归零、不阻断）
        if l2_enabled:
            try:
                await maybe_update_l2(
                    memory_store,
                    agent_id,
                    llm,
                    current_l1=summary_text,
                    every_merges=l2_every_merges,
                    max_chars=l2_max_chars,
                )
            except Exception as exc:  # noqa: BLE001 - L2 失败不得阻断回复（D8）
                logger.warning("L2 重合成失败（agent=%s）：%s", agent_id, exc)

        removals = [RemoveMessage(id=m.id) for m in old_chunk if m.id]
        return {"messages": removals} if removals else {}
    except Exception as exc:  # noqa: BLE001 - 摘要失败不得阻断记忆写入（D8）
        logger.warning(
            "滚动摘要失败，本轮记忆已写入、缓冲不裁剪，下轮重试（agent=%s）：%s",
            agent_id,
            exc,
        )
        return {}
