"""节点：调用 LLM 流式生成回复，并把 token 透出给服务层推 SSE（M3 / M5 / M6）。

方案 3.7：`generate` 节点逐 token 产出；流式下经 LangGraph `StreamWriter` 透出
（服务层用 `stream_mode="custom"` 收取）。非流式（`ainvoke`）下 writer 为框架的
no-op，token 被静默丢弃，`reply` 仍由增量拼接得到，行为与 M1 一致。

M5：角色可一次连发 1~max_messages 条消息，用 `MESSAGE_DELIMITER` 分隔。本节点在流式
过程中按分隔符切分：透出 `{"text": ...}`（增量文本）与 `{"break": ...}`（分条边界），
服务层据此在界面上新起气泡。图状态里每条消息落一条 AIMessage，保证
「1 条消息 = 1 条记忆 = 1 行快照 = 1 条工作记忆」严格一致（撤回时按内容精确命中）。

M6：每条发言附带一段「内心独白」（思考链），采用**交错格式**：消息内容 → `THINK_DELIMITER`
→ 该条内心独白 → `MESSAGE_DELIMITER` → 下一条消息……本节点在流式过程中按标记切分，进入
思考态后的文本**不再透出**（只收集为 `thinkings`），因此思考不会出现在流式气泡里；思考态
遇到下一个 `MESSAGE_DELIMITER` 会切回消息态，每条消息与其内心独白天然成对。缺失时降级为空
串（前端不显示按钮）。思考**不进** `messages`（不写入记忆/工作记忆/群聊 fanout），仅作快照
附加列由服务层落库。条数由服务层 `normalize_thinkings` 兜底对齐。
"""

import json
import logging
import re
import uuid

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from app.agents.state import RolePlayState
from app.llm.base import LLMProvider
from app.lore.chunker import estimate_tokens
from app.observability.trace import is_degraded
from app.prompts.roleplay_system import MESSAGE_DELIMITER, THINK_DELIMITER, THINK_MAX_CHARS
from app.skills.executor import MAX_TOTAL_TOKENS
from app.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# 空输出重试时补的一条 user 提示：模型偶发只输出内心独白/空串（尤其闲聊轮 prompt
# 以 assistant 收尾时）。要求「正文 + 内心独白」完整格式，避免重试文本又丢思考。
_EMPTY_RETRY_NUDGE = (
    "（请重新完整输出：先写你要说的消息正文，另起一行单独输出 {think}，"
    "再另起一行写这条消息的内心独白（不超过 {limit} 字）；正文不能为空，内心独白不能省略。）"
)
# 内心独白缺失时的补写提示：正文意思保持不变，只补上每条消息的独白，避免前端时有时无。
_THINK_RETRY_NUDGE = (
    "（你刚才漏掉了每条消息的内心独白。请保持正文内容不变、按格式重新输出："
    "先写消息正文，另起一行单独输出 {think}，再另起一行写这条消息的内心独白"
    "（不超过 {limit} 字）；每条消息都必须有内心独白，不能省略。）"
)


def _nudge(template: str) -> str:
    """把重试提示模板里的分隔符/上限占位符填好。"""
    return template.format(think=THINK_DELIMITER, limit=THINK_MAX_CHARS)


# 用户明确要求某工具、模型却只回文本时的一次强制补调提示（避免“说设好了但抽屉里没有”）
_TOOL_RETRY_NUDGE = "（你刚才没有调用 {tool} 工具。请现在调用它完成任务；只有工具返回 ok=true 时才能说已完成。）"
# 用户话语 → 工具意图。带「明确钟点」的提醒才判定为用户在要求创建提醒，
# 避免「提醒我一点小事」「还有一点没做完」这类闲聊被误判。
# 钟点特征：① 21:30 ② 8点35 / 七点半 / 十点整 / 九点了 ③ 今晚8点 / 明天早上七点
_CLOCK_RE = re.compile(
    r"\d{1,2}\s*[:：]\s*\d{2}"
    r"|[零一二三四五六七八九十两\d]+\s*[点时]\s*(半|整|刻|[0-5]?\d\s*分?|了)"
    r"|(早上|上午|中午|下午|晚上|今晚|明早|明天|凌晨|半夜)\s*[零一二三四五六七八九十两\d]+\s*[点时]"
)


def _stream_writer():
    """取当前图运行的 StreamWriter；不在图运行上下文（如单测直接调用）时返回 None。"""
    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 - 图外无 runtime，降级为无流式
        return None


def normalize_thinkings(
    thinkings: list[str] | None, n: int, *, limit: int = THINK_MAX_CHARS
) -> list[str]:
    """把思考列表对齐到 n 条：每条截断到 limit 字、丢弃多余、不足补空串。"""
    out = [str(t).strip()[:limit] for t in (thinkings or [])][:n]
    out += [""] * (n - len(out))
    return out


def _first_line(text: str) -> str:
    """取内心独白的第一行非空内容（截断到上限）；无内容时返回空串。"""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s[:THINK_MAX_CHARS]
    return ""


def split_messages(text: str, *, max_messages: int = 3) -> list[str]:
    """把完整文本按分隔符切成 1~max_messages 条（去空白、去空条）。

    超过上限的剩余部分合并进最后一条（不丢弃内容）。全空时回退为空串单条。
    """
    parts = [p.strip() for p in text.split(MESSAGE_DELIMITER)]
    parts = [p for p in parts if p]
    if not parts:
        return [""]
    if len(parts) > max_messages:
        head = parts[: max_messages - 1]
        tail = "\n".join(parts[max_messages - 1 :])
        parts = head + [tail]
    return parts


def parse_replies(text: str, *, max_messages: int = 3) -> tuple[list[str], list[str]]:
    """把一段完整文本按交错格式解析为 (正文列表, 思考列表)。

    与流式切分等价：先按 `MESSAGE_DELIMITER` 分段，每段内按 `THINK_DELIMITER`
    切出「正文 + 该条内心独白（取首行）」。丢弃正文为空的分段；超过上限的多余分段
    合并进最后一条（用 `\\n` 连接，思考取该组首条）。供空输出重试等非流式场景复用。
    """
    pairs: list[tuple[str, str]] = []
    for seg in text.split(MESSAGE_DELIMITER):
        content, _, think = seg.partition(THINK_DELIMITER)
        content = content.strip()
        if content:
            pairs.append((content, _first_line(think)))
    if not pairs:
        return [], []
    if len(pairs) > max_messages:
        head = pairs[: max_messages - 1]
        rest = pairs[max_messages - 1 :]
        merged = ("\n".join(c for c, _ in rest), rest[0][1])
        pairs = head + [merged]
    return [c for c, _ in pairs], [t for _, t in pairs]


async def _retry_for_thinkings(
    state: RolePlayState,
    llm: LLMProvider,
    replies: list[str],
    thinkings: list[str],
    *,
    max_messages: int,
    enabled: bool = True,
) -> tuple[list[str], list[str]]:
    """本轮所有消息都缺内心独白时，带格式提醒重出一次；失败则保持原样。

    内心独白完全依赖模型自觉，实测约七成回合会被省略，导致前端「思考」按钮时有时无。
    这里做一次确定性校验：只有「一条独白都没有」才补写（已有独白不折腾），且要求补写
    结果的条数与原文一致才采纳，避免正文与独白错位。

    `enabled=False` 时完全不触发（省掉这次额外调用；`REPLY_THINK_RETRY_ENABLED`）。
    """
    if not enabled:
        return replies, thinkings
    if not replies or not any(r.strip() for r in replies):
        return replies, thinkings
    if any(t for t in thinkings):
        return replies, thinkings
    prompt = state.get("prompt") or []
    if not prompt:
        return replies, thinkings
    logger.info(
        "think-retry fired (replies=%d, chars=%d)",
        len(replies),
        sum(len(r) for r in replies),
    )
    try:
        # 必须把「刚才的回复」作为 assistant 回合带回去：否则模型看不到自己写了什么，
        # 只会重新回答一遍用户的话，多半仍然不写内心独白。
        text = await llm.generate(
            [
                *prompt,
                {"role": "assistant", "content": "\n".join(replies)},
                {"role": "user", "content": _nudge(_THINK_RETRY_NUDGE)},
            ]
        )
    except Exception:  # noqa: BLE001 - 补写失败不改变原流程，沿用空独白降级
        logger.warning("内心独白补写失败", exc_info=True)
        return replies, thinkings
    new_replies, new_thinkings = parse_replies(text, max_messages=max_messages)
    if not any(new_thinkings):
        logger.info("内心独白补写未采纳：模型仍未输出独白（replies=%d）", len(new_replies))
        return replies, thinkings
    if len(new_replies) == len(replies):
        return new_replies, new_thinkings
    # 条数不一致：保留原正文，只把补写的独白按顺序对齐（避免正文被改写/错位）
    return replies, normalize_thinkings(new_thinkings, len(replies))


async def generate(
    state: RolePlayState,
    llm: LLMProvider,
    *,
    max_messages: int = 3,
    skill_registry: SkillRegistry | None = None,
    max_tool_rounds: int = 3,
    allowed_skills: list[str] | None = None,
    think_retry_enabled: bool = True,
) -> dict:
    """生成节点入口（V2.2 契约 §2.4）。

    白名单为空 / 无注册表 → 原 `astream` 路径（行为与 V2.1 逐字一致）；
    否则走工具模式：`generate_with_tools` 请求，可能只回 tool_calls（不写空 AIMessage）。
    """
    allowed = list(state.get("allowed_skills") or allowed_skills or [])
    if skill_registry is not None and allowed:
        result = await _generate_with_tools(
            state,
            llm,
            skill_registry,
            allowed=allowed,
            max_messages=max_messages,
            max_tool_rounds=max_tool_rounds,
            think_retry_enabled=think_retry_enabled,
        )
    else:
        result = await _generate_stream(
            state, llm, max_messages=max_messages, think_retry_enabled=think_retry_enabled
        )
    # V2.4：LLM 超时兜底时把降级标记写进状态，update_memory 据此跳过所有记忆写入
    if is_degraded():
        result["degraded"] = True
    return result


async def _generate_stream(
    state: RolePlayState,
    llm: LLMProvider,
    *,
    max_messages: int = 3,
    think_retry_enabled: bool = True,
) -> dict:
    writer = _stream_writer()
    # 本轮上限可被状态覆盖（群聊闲聊每角色 ≤2 条）；0/缺省回退到图构造时的默认值。
    limit = state.get("reply_max_messages") or max_messages
    max_messages = max(1, int(limit))
    msg = MESSAGE_DELIMITER
    think = THINK_DELIMITER
    # 回退缓冲：需同时防止「消息分隔符」与「思考标记」被拆分在两个增量之间，
    # 故取两者长度的较大者 - 1（残缺标记必须先留在 buffer，不能当作正文透出）。
    keep = max(len(msg), len(think)) - 1

    buf = ""
    mode = "msg"  # "msg" 收集消息并透出；"think" 只收集思考、不透出
    current = ""
    current_think = ""
    parts: list[str] = []
    thinks: list[str] = []
    breaks = 0

    def emit(text: str) -> None:
        nonlocal current
        if text:
            current += text
            if writer is not None:
                writer({"text": text})

    def close_message() -> None:
        """封一条：正文入 parts、其内心独白（首行）入 thinks，并清空当前缓冲。"""
        nonlocal current, current_think, breaks
        parts.append(current)
        thinks.append(_first_line(current_think))
        current = ""
        current_think = ""
        breaks += 1

    def do_break() -> None:
        if writer is not None:
            writer({"break": True})

    async for delta in llm.astream(state["prompt"]):  # type: ignore[arg-type]
        buf += delta
        while buf:
            if mode == "msg":
                ti = buf.find(think)
                mi = buf.find(msg)
                if ti == -1 and mi == -1:
                    if len(buf) <= keep:
                        break  # 尾部可能是残缺标记，留待下个增量
                    emit(buf[:-keep])
                    buf = buf[-keep:]
                    break
                if ti != -1 and (mi == -1 or ti < mi):
                    # 当前消息正文结束，转入只收集思考的 mode（不再透出）
                    emit(buf[:ti])
                    buf = buf[ti + len(think) :]
                    mode = "think"
                    continue
                # 命中消息分隔符：封一条并新起一条（达上限则丢弃该边界、并入当前条）
                emit(buf[:mi])
                buf = buf[mi + len(msg) :]
                if breaks < max_messages - 1:
                    close_message()
                    do_break()
                continue
            else:  # think：绝不 writer 透出；遇消息分隔符则切回消息态
                mi = buf.find(msg)
                if mi == -1:
                    if len(buf) <= keep:
                        break  # 尾部可能是残缺标记，保留在 buffer 里等待下个增量
                    current_think += buf[:-keep]
                    buf = buf[-keep:]
                    break
                current_think += buf[:mi]
                buf = buf[mi + len(msg) :]
                if breaks < max_messages - 1:
                    close_message()
                    do_break()
                mode = "msg"
                continue

    # 收尾：msg 态吐出残余；think 态并入当前思考，最后再封一条
    if mode == "think":
        current_think += buf
    else:
        emit(buf)
    close_message()

    # 过滤空消息：用 zip 保证「消息 ↔ 思考」始终成对，避免错位
    replies: list[str] = []
    thinkings: list[str] = []
    for raw, th in zip(parts, thinks):
        text = raw.strip()
        if text:
            replies.append(text)
            thinkings.append(th)
    if not replies:
        # 空输出兜底重试：模型偶发只输出内心独白/空串（实测闲聊轮 prompt 以 assistant
        # 收尾时必现）。此时流式无正文透出，补一条 user 提示做一次非流式重试即可纠正；
        # 重试文本由 `done` 事件回填前端展示，无需重放流式增量。
        try:
            retry_text = await llm.generate(
                [*state["prompt"], {"role": "user", "content": _nudge(_EMPTY_RETRY_NUDGE)}]
            )
            replies, thinkings = parse_replies(retry_text, max_messages=max_messages)
        except Exception:  # noqa: BLE001 - 重试失败不改变原流程，沿用空回复兜底
            logger.warning("空输出重试失败", exc_info=True)
            replies, thinkings = [], []
    if not replies:
        replies, thinkings = [""], [""]
    # 校验：一条独白都没有 → 带格式提醒补写一次（补写失败则静默沿用空独白）
    replies, thinkings = await _retry_for_thinkings(
        state,
        llm,
        replies,
        thinkings,
        max_messages=max_messages,
        enabled=think_retry_enabled,
    )
    thinkings = normalize_thinkings(thinkings, len(replies))

    # 自主闲聊：给每条 AIMessage 打 idle 标记，供 update_memory 摘要时排除、清空时定位。
    kwargs = {"additional_kwargs": {"idle": True}} if state.get("idle_scope") else {}
    return {
        "reply": "\n".join(replies),
        "replies": replies,
        "thinkings": thinkings,
        "messages": [AIMessage(content=r, **kwargs) for r in replies],
    }


# ===== V2.2 工具模式（契约 §2.4）=====


def invalid_argument_streak(results: list[dict] | None) -> int:
    """同工具连续 `invalid_arguments` 次数（连续 2 次 → 强制文本，§2.6）。"""
    streak = 0
    name: str | None = None
    for record in reversed(list(results or [])):
        if record.get("error_type") != "invalid_arguments":
            break
        if name is None:
            name = record.get("name")
            streak = 1
        elif record.get("name") == name:
            streak += 1
        else:
            break
    return streak


def _turn_tool_results(state: RolePlayState) -> list[dict]:
    """只取本轮（同一 `tool_turn`）的工具结果。

    `tool_results` 是 `operator.add` 累积字段且经 checkpointer 跨轮持久化；若不过滤，
    上一轮的记录会一直留在状态里，导致 `tool_rounds` 与 `invalid_arguments` 判定被
    永久占满（会话内再也无法调用任何工具）。`tool_turn` 为空时（旧调用/单测直接构造
    状态）退回全量，保持兼容。
    """
    results = list(state.get("tool_results") or [])
    turn = state.get("tool_turn") or ""
    if not turn:
        return results
    return [r for r in results if r.get("tool_turn") == turn]


def _tool_result_content(record: dict) -> str:
    return json.dumps(
        {
            "ok": record.get("ok"),
            "data": record.get("data"),
            "error_type": record.get("error_type"),
            "message": record.get("message", ""),
        },
        ensure_ascii=False,
        default=str,
    )


def _build_tool_messages(state: RolePlayState, *, budget: int = MAX_TOTAL_TOKENS) -> list[dict]:
    """重建本地 LLM 消息列表：prompt + 各轮 assistant(tool_calls) + tool 结果。

    工具消息**只存在于这里**，不写入 `state.messages`（契约 §2.4）；总量超 8000 token
    时按时间顺序丢弃最早的结果，保留最近（§2.7）。
    """
    messages: list[dict] = [dict(m) for m in (state.get("prompt") or [])]
    groups: dict[int, list[dict]] = {}
    for record in _turn_tool_results(state):
        groups.setdefault(int(record.get("tool_round") or 0), []).append(record)
    ordered = [groups[key] for key in sorted(groups)]

    kept: list[list[dict]] = []
    used = 0
    for group in reversed(ordered):
        cost = sum(estimate_tokens(_tool_result_content(r)) for r in group)
        if kept and used + cost > budget:
            break
        kept.append(group)
        used += cost
    kept.reverse()

    for group in kept:
        calls = [
            {
                "id": r.get("tool_call_id"),
                "type": "function",
                "function": {
                    "name": r.get("name"),
                    "arguments": json.dumps(r.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for r in group
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        for record in group:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": record.get("tool_call_id"),
                    "content": _tool_result_content(record),
                }
            )
    return messages


def _emit_final_text(writer, replies: list[str]) -> None:  # noqa: ANN001
    """工具模式：最终文本一次性下发（不 token 级流式，契约 D1）。"""
    text = "\n".join(r for r in replies if r)
    if writer is not None and text:
        writer({"text": text})


def _reply_payload(state: RolePlayState, replies: list[str], thinkings: list[str]) -> dict:
    kwargs = {"additional_kwargs": {"idle": True}} if state.get("idle_scope") else {}
    return {
        "reply": "\n".join(replies),
        "replies": replies,
        "thinkings": thinkings,
        "messages": [AIMessage(content=r, **kwargs) for r in replies],
    }


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _explicit_tool_intent(messages: list[dict], allowed: list[str]) -> str | None:
    """用户本轮是否明确点名了某个带副作用的工具（提醒需带具体钟点）。

    仅在「本轮没有任何工具结果」时调用；命中后上层用 `tool_choice` 强制补调一次，
    保证「说设好了但抽屉里没有」不再发生。
    """
    text = _last_user_text(messages).lower()
    if (
        "set_reminder" in allowed
        and ("提醒" in text or "remind" in text)
        and _CLOCK_RE.search(text)
    ):
        return "set_reminder"
    if "roll_dice" in allowed and any(word in text for word in ("掷骰", "骰子", "roll")):
        return "roll_dice"
    return None


async def _empty_reply_fallback(
    state: RolePlayState,
    llm: LLMProvider,
    *,
    max_messages: int,
    think_retry_enabled: bool = True,
) -> dict:
    """空输出兜底重试（与流式路径同一策略）：补一条 user 提示做一次非流式重试。"""
    try:
        retry_text = await llm.generate(
            [
                *(state.get("prompt") or []),
                {"role": "user", "content": _nudge(_EMPTY_RETRY_NUDGE)},
            ]
        )
        replies, thinkings = parse_replies(retry_text, max_messages=max_messages)
    except Exception:  # noqa: BLE001 - 重试失败不改变原流程，沿用空回复兜底
        logger.warning("工具模式空输出重试失败", exc_info=True)
        replies, thinkings = [], []
    if not replies:
        replies, thinkings = [""], [""]
    # 校验：一条独白都没有 → 带格式提醒补写一次（补写失败则静默沿用空独白）
    replies, thinkings = await _retry_for_thinkings(
        state,
        llm,
        replies,
        thinkings,
        max_messages=max_messages,
        enabled=think_retry_enabled,
    )
    thinkings = normalize_thinkings(thinkings, len(replies))
    _emit_final_text(_stream_writer(), replies)
    return _reply_payload(state, replies, thinkings)


async def _generate_with_tools(
    state: RolePlayState,
    llm: LLMProvider,
    registry: SkillRegistry,
    *,
    allowed: list[str],
    max_messages: int,
    max_tool_rounds: int,
    think_retry_enabled: bool = True,
) -> dict:
    tools = registry.to_openai_tools(allowed)
    turn_results = _turn_tool_results(state)
    tool_rounds = int(state.get("tool_rounds") or 0)
    forced = tool_rounds >= max_tool_rounds or invalid_argument_streak(turn_results) >= 2
    messages = _build_tool_messages(state)

    try:
        reply = await llm.generate_with_tools(
            messages, tools, tool_choice="none" if forced else "auto", temperature=0.0
        )
    except NotImplementedError:
        # provider 不支持 tools → 降级为无工具文本回答（契约 §2.3）
        out = await _generate_stream(state, llm, max_messages=max_messages)
        out["tools_unavailable"] = True
        return out

    if reply.tool_calls and not forced:
        # 只回工具调用：不写空 AIMessage；tool_rounds 保持当前（由 tools 节点 +1）
        return {
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.tool_calls
            ],
            "tool_rounds": tool_rounds,
            "reply": "",
            "replies": [],
            "thinkings": [],
            "messages": [],
        }

    if reply.tool_calls and forced:
        # 轮数用尽/连续参数错仍请求工具 → 丢弃，记 loop_detected，走空回复兜底（§2.4-4）
        loops = [
            {
                "tool_call_id": c.id,
                "name": c.name,
                "arguments": c.arguments,
                "ok": False,
                "error_type": "loop_detected",
                "message": "已达工具轮数上限，强制文本收尾",
                "latency_ms": 0,
                "cached": False,
                "tool_round": tool_rounds,
                "tool_turn": state.get("tool_turn") or "",
            }
            for c in reply.tool_calls
        ]
        payload = await _empty_reply_fallback(
            state,
            llm,
            max_messages=max_messages,
            think_retry_enabled=think_retry_enabled,
        )
        payload["tool_results"] = loops
        return payload

    # 文本收尾（无 tool_calls）
    replies, thinkings = parse_replies(reply.content or "", max_messages=max_messages)
    if not replies:
        return await _empty_reply_fallback(
            state,
            llm,
            max_messages=max_messages,
            think_retry_enabled=think_retry_enabled,
        )

    # 用户明确要求工具、模型却只回文本（本轮无任何工具结果）→ 强制补调一次。
    # 这是“说设好了但提醒抽屉里没有”的确定性兜底：用指定函数的 tool_choice 锁死工具，
    # 模型只能调用或明确失败，不会再用客套话糊弄过去。
    hint = None
    if not turn_results:
        hint = _explicit_tool_intent(messages, allowed)
    if hint is not None:
        retry = None
        try:
            retry = await llm.generate_with_tools(
                [
                    *messages,
                    {"role": "assistant", "content": reply.content or ""},
                    {"role": "user", "content": _TOOL_RETRY_NUDGE.format(tool=hint)},
                ],
                tools,
                tool_choice={"type": "function", "function": {"name": hint}},
                temperature=0.0,
            )
        except Exception:  # noqa: BLE001 - 补调失败不改变原流程，沿用首轮文本
            logger.warning("工具补调失败", exc_info=True)
        if retry is not None and retry.tool_calls:
            return {
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in retry.tool_calls
                ],
                "tool_rounds": tool_rounds,
                "reply": "",
                "replies": [],
                "thinkings": [],
                "messages": [],
            }
        if retry is not None and (retry.content or "").strip():
            retries, retry_thinks = parse_replies(
                retry.content, max_messages=max_messages
            )
            if retries:
                replies = retries
                thinkings = normalize_thinkings(retry_thinks, len(retries))

    # 校验：一条独白都没有 → 带格式提醒补写一次（补写失败则静默沿用空独白）
    replies, thinkings = await _retry_for_thinkings(
        state,
        llm,
        replies,
        thinkings,
        max_messages=max_messages,
        enabled=think_retry_enabled,
    )
    thinkings = normalize_thinkings(thinkings, len(replies))
    _emit_final_text(_stream_writer(), replies)
    return _reply_payload(state, replies, thinkings)
