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

import logging

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from app.agents.state import RolePlayState
from app.llm.base import LLMProvider
from app.prompts.roleplay_system import MESSAGE_DELIMITER, THINK_DELIMITER, THINK_MAX_CHARS

logger = logging.getLogger(__name__)

# 空输出重试时补的一条 user 提示：模型偶发只输出内心独白/空串（尤其闲聊轮 prompt
# 以 assistant 收尾时）。该提示仅用于本次重试调用，不写入状态/记忆/快照。
_EMPTY_RETRY_NUDGE = "（请只输出你要说的话；务必先写出消息正文，不要只输出内心独白。）"


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


async def generate(state: RolePlayState, llm: LLMProvider, *, max_messages: int = 3) -> dict:
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
                [*state["prompt"], {"role": "user", "content": _EMPTY_RETRY_NUDGE}]
            )
            replies, thinkings = parse_replies(retry_text, max_messages=max_messages)
        except Exception:  # noqa: BLE001 - 重试失败不改变原流程，沿用空回复兜底
            logger.warning("空输出重试失败", exc_info=True)
            replies, thinkings = [], []
    if not replies:
        replies, thinkings = [""], [""]
    thinkings = normalize_thinkings(thinkings, len(replies))

    # 自主闲聊：给每条 AIMessage 打 idle 标记，供 update_memory 摘要时排除、清空时定位。
    kwargs = {"additional_kwargs": {"idle": True}} if state.get("idle_scope") else {}
    return {
        "reply": "\n".join(replies),
        "replies": replies,
        "thinkings": thinkings,
        "messages": [AIMessage(content=r, **kwargs) for r in replies],
    }
