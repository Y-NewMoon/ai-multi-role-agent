"""群聊调度（M4 · ADR-0004 §3.4 / §3.7）。

V1 采用 **LLM supervisor + 硬上限**：
- LLM 决定下一位发言人（返回角色编号或 END）；
- 硬约束（不依赖 LLM 自觉）：一轮内每人最多发言一次、连续同人强制换人、
  一轮最多 `max_turns` 次发言、解析失败退化为按成员顺序轮询（永不卡死）。
- 一轮内**串行**调用各角色自己的图实例（接缝 3：每角色一张图）。

本轮上下文注入（实现细节，ADR 未明确）：除按 ADR §3.2 显式注入用户消息外，
还把**本轮更早发言者的公屏回复**以 assistant 消息（内容带 `【角色名】` 前缀）注入
下一位发言者的消息列表，否则同轮角色之间互相"看不见"，群聊不成立。
（用户消息作为最后一条 human 消息，确保 `update_memory` 仍正确取到本轮用户发言。）

M5：一个角色一轮可连发多条（`turn_done.replies`）。注入他人回复时**逐条**注入
（每条都带 `【角色名】` 前缀），保证撤回时能按内容精确命中各成员 thread。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.nodes.generate import normalize_thinkings
from app.llm.base import LLMProvider

_DECISION_SYSTEM = (
    "你是一个多人聊天室的主持人。根据群成员与你眼前的对话，判断下一位应当发言的成员编号。"
    "当本轮讨论已充分、或没有成员需要继续发言时，输出 END。"
    "严格只输出一个成员编号（纯数字）或 END，不要输出任何解释。"
)


@dataclass(frozen=True)
class GroupMember:
    """群成员的角色档案（服务层预取，图内只读，接缝 5）。"""

    agent_id: int
    name: str
    persona_prompt: str
    background: str


_PREFIX_RE = re.compile(r"^\s*【([^】]{1,20})】\s*")


def strip_speaker_prefix(text: str, names: set[str]) -> str:
    """剥离回复开头模型自带的「【角色名】」前缀（仅当名字属本群成员时）。

    群聊注入他人发言时带 `【角色名】` 前缀，模型会照此模仿、把前缀（甚至他人台词）写进
    自己的输出。这里做兜底剥离，保证落库/展示/工作记忆三处内容一致。剥离后若为空则保留
    原文，避免整条消息被清空；连续多个前缀（如「【A】【B】正文」）会逐个剥离。
    """
    out = text
    while True:
        m = _PREFIX_RE.match(out)
        if m is None or m.group(1).strip() not in names:
            break
        stripped = out[m.end() :]
        if not stripped.strip():
            break
        out = stripped
    return out


def parse_decision(raw: str, valid_ids: set[int]) -> tuple[str, int | None]:
    """解析 supervisor 的 LLM 决策：("speak", id) / ("end", None) / ("invalid", None)。"""
    text = (raw or "").strip()
    if "END" in text.upper():
        return "end", None
    match = re.search(r"\d+", text)
    if match is not None:
        value = int(match.group())
        if value in valid_ids:
            return "speak", value
    return "invalid", None


def choose_next(
    *,
    candidate: int | None,
    order: list[int],
    spoken: list[int],
    last_speaker: int | None,
    max_turns: int,
) -> int | None:
    """在硬约束下选出下一位发言人；返回 None 表示本轮结束。

    - 已达 max_turns 或全部成员都已发言 → 结束；
    - LLM 选的人无效 / 已发言 / 与上一位相同 → 退化为按 order 轮询（保证换人）。
    """
    if len(spoken) >= max_turns:
        return None
    unspoken = [a for a in order if a not in spoken]
    if not unspoken:
        return None
    if candidate is not None and candidate in unspoken and candidate != last_speaker:
        return candidate
    return unspoken[0]  # unspoken 必不含 last_speaker（其已在 spoken 中）


class Supervisor:
    """一次群聊轮次的调度器：持有各角色的图实例缓存。"""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        graph_factory: Callable[[], Any],
        members: list[GroupMember],
        max_turns: int = 4,
    ) -> None:
        self._llm = llm
        self._graph_factory = graph_factory
        self._members = list(members)
        self._order = [m.agent_id for m in self._members]
        self._by_id = {m.agent_id: m for m in self._members}
        self._valid_ids = set(self._order)
        self._names = {m.name for m in self._members}
        self._max_turns = max_turns
        self._graphs: dict[int, Any] = {}

    def _graph(self, agent_id: int) -> Any:
        """每角色一张图实例（接缝 3），按需创建并缓存。"""
        if agent_id not in self._graphs:
            self._graphs[agent_id] = self._graph_factory()
        return self._graphs[agent_id]

    def _transcript_text(
        self, user_message: str, transcript: list[tuple[GroupMember, list[str]]]
    ) -> str:
        roster = "、".join(f"{m.agent_id}={m.name}" for m in self._members)
        lines = [f"群成员：{roster}", f"用户说：{user_message}"]
        if transcript:
            lines.append("本轮已发言：")
            lines.extend(f"  {m.agent_id}={m.name}：{' '.join(parts)}" for m, parts in transcript)
        else:
            lines.append("本轮尚未有人发言。")
        return "\n".join(lines)

    async def _decide(
        self, user_message: str, transcript: list[tuple[GroupMember, list[str]]]
    ) -> tuple[str, int | None]:
        raw = await self._llm.generate(
            [
                {"role": "system", "content": _DECISION_SYSTEM},
                {"role": "user", "content": self._transcript_text(user_message, transcript)},
            ]
        )
        return parse_decision(raw, self._valid_ids)

    def _state_in(
        self,
        member: GroupMember,
        group_session_id: int,
        user_message: str,
        transcript: list[tuple[GroupMember, list[str]]],
    ) -> dict:
        messages: list[Any] = [HumanMessage(content=user_message)]
        for m, parts in transcript:
            messages.extend(AIMessage(content=f"【{m.name}】{p}") for p in parts)
        return {
            "messages": messages,
            "agent_id": member.agent_id,
            "session_id": f"g{group_session_id}",  # 摘要 key = summary:g{id}（ADR-0004 §3.6）
            "agent_name": member.name,
            "persona_prompt": member.persona_prompt,
            "background": member.background,
            "idle_scope": "",  # 显式置空，避免上一次闲聊残留标记污染正常轮
            "reply_max_messages": 0,  # 正常轮用图默认上限
        }

    def _idle_state_in(
        self,
        member: GroupMember,
        group_session_id: int,
        transcript: list[tuple[GroupMember, list[str]]],
        reply_max_messages: int,
    ) -> dict:
        """自主闲聊的图状态输入：无用户消息，本轮他人闲聊以带 idle 标记的 AIMessage 注入。"""
        messages: list[Any] = []
        for m, parts in transcript:
            messages.extend(
                AIMessage(content=f"【{m.name}】{p}", additional_kwargs={"idle": True})
                for p in parts
            )
        return {
            "messages": messages,
            "agent_id": member.agent_id,
            "session_id": f"g{group_session_id}",
            "agent_name": member.name,
            "persona_prompt": member.persona_prompt,
            "background": member.background,
            "idle_scope": "group",
            "reply_max_messages": reply_max_messages,
        }

    async def run(
        self, *, group_session_id: int, user_message: str
    ) -> AsyncIterator[tuple[str, dict]]:
        """产出一轮的事件：`speaker` → (`token` / `message_break`)* → `turn_done`（每个发言人一组）。"""
        spoken: list[int] = []
        last_speaker: int | None = None
        transcript: list[tuple[GroupMember, list[str]]] = []

        while True:
            decision, candidate = await self._decide(user_message, transcript)
            if decision == "end" and spoken:
                break  # LLM 明确结束，且本轮已有人发言
            # 首轮 LLM 若直接 END（或解析失败）→ 走轮询兜底，保证用户至少收到一个回复
            nxt = choose_next(
                candidate=candidate if decision == "speak" else None,
                order=self._order,
                spoken=spoken,
                last_speaker=last_speaker,
                max_turns=self._max_turns,
            )
            if nxt is None:
                break

            member = self._by_id[nxt]
            yield "speaker", {"agent_id": nxt, "name": member.name}

            replies: list[str] = []
            thinkings: list[str] = []
            reply = ""
            graph = self._graph(nxt)
            async for mode, chunk in graph.astream(
                self._state_in(member, group_session_id, user_message, transcript),
                config={"configurable": {"thread_id": f"g{group_session_id}:{nxt}"}},
                stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    if isinstance(chunk, dict) and chunk.get("break"):
                        yield "message_break", {"agent_id": nxt}
                    else:
                        text = str(chunk.get("text", "")) if isinstance(chunk, dict) else ""
                        if text:
                            yield "token", {"agent_id": nxt, "text": text}
                elif mode == "values":
                    # 每步状态快照；generate 完成后带 replies（回退到 reply 以兼容旧图）
                    if chunk.get("replies"):
                        replies = [str(r) for r in chunk["replies"]]
                    if chunk.get("reply") is not None:
                        reply = str(chunk["reply"])
                    if chunk.get("thinkings") is not None:
                        thinkings = [str(t) for t in chunk["thinkings"]]

            if not replies:
                replies = [reply]
            replies = [strip_speaker_prefix(r, self._names) for r in replies]
            yield "turn_done", {
                "agent_id": nxt,
                "replies": replies,
                "reply": "\n".join(replies),
                "thinkings": normalize_thinkings(thinkings, len(replies)),
            }
            spoken.append(nxt)
            last_speaker = nxt
            transcript.append((member, replies))

    # ===== 自主闲聊（M6：用户点击按钮触发一次）=====

    def _rotate(self, after: int | None) -> list[int]:
        """按「上一位发言人」把成员顺序轮转，使上一位排到末尾、其下一位优先开口。

        公平轮转：多次点击会从不同成员开始，避免总是同一两人包场（用户要求）。
        `after` 为空或不在群内时按原顺序。
        """
        order = list(self._order)
        if after is None or after not in order:
            return order
        i = order.index(after)
        return order[i + 1 :] + order[: i + 1]

    async def run_idle(
        self,
        *,
        group_session_id: int,
        max_total: int = 10,
        turn_max_messages: int = 2,
        rotate_after: int | None = None,
    ) -> AsyncIterator[tuple[str, dict]]:
        """产出一轮自主闲聊的事件：`speaker` → (`token` / `message_break`)* → `turn_done`。

        与 `run` 的区别：无用户消息；按成员顺序各发一次（上一位发言人排到末尾以轮转），
        每人最多 `turn_max_messages` 条，累计不超过 `max_total` 条；每条回复都带 idle 标记。
        """
        transcript: list[tuple[GroupMember, list[str]]] = []
        total = 0

        for agent_id in self._rotate(rotate_after):
            if total >= max_total:
                break
            member = self._by_id[agent_id]
            yield "speaker", {"agent_id": agent_id, "name": member.name}
            replies: list[str] = []
            thinkings: list[str] = []
            reply = ""
            graph = self._graph(agent_id)
            async for mode, chunk in graph.astream(
                self._idle_state_in(member, group_session_id, transcript, turn_max_messages),
                config={"configurable": {"thread_id": f"g{group_session_id}:{agent_id}"}},
                stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    if isinstance(chunk, dict) and chunk.get("break"):
                        yield "message_break", {"agent_id": agent_id}
                    else:
                        text = str(chunk.get("text", "")) if isinstance(chunk, dict) else ""
                        if text:
                            yield "token", {"agent_id": agent_id, "text": text}
                elif mode == "values":
                    if chunk.get("replies"):
                        replies = [str(r) for r in chunk["replies"]]
                    if chunk.get("reply") is not None:
                        reply = str(chunk["reply"])
                    if chunk.get("thinkings") is not None:
                        thinkings = [str(t) for t in chunk["thinkings"]]

            if not replies:
                replies = [reply]
            replies = [strip_speaker_prefix(r, self._names) for r in replies]
            # 每人不超 turn_max_messages；再按累计预算截断（保证总量 ≤ max_total）
            remaining = max_total - total
            parts = replies[:turn_max_messages][:remaining]
            if not parts:
                parts = [replies[0] if replies else ""]
            total += len(parts)
            yield "turn_done", {
                "agent_id": agent_id,
                "replies": parts,
                "reply": "\n".join(parts),
                # 思考按被保留的消息条数同步截断并对齐
                "thinkings": normalize_thinkings(thinkings, len(parts)),
            }
            transcript.append((member, parts))
