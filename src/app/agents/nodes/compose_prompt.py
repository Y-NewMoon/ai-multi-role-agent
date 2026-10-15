"""节点：拼装发给 LLM 的消息（system + 摘要 + 召回记忆 + 近 N 轮）。

方案 3.9：总预算 8k token（可配），按优先级裁剪：
`system(persona)`（必留）> 近 N 轮原文 > 召回记忆 top-k > 滚动摘要
—— 越低优先级越先被丢弃。
"""

from langchain_core.messages import BaseMessage

from app.agents.state import RolePlayState
from app.prompts.roleplay_system import build_idle_instruction, build_idle_nudge

_ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}


def _to_chat_messages(messages: list[BaseMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = _ROLE_MAP.get(getattr(m, "type", "human"), "user")
        out.append({"role": role, "content": str(m.content)})
    return out


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf"


def _recent_rounds(messages: list[BaseMessage], rounds: int) -> list[BaseMessage]:
    """取最近 rounds 轮（一轮 = 一条用户消息 + 其后的角色回复）。

    M5：一轮可能含多条角色回复，故按「轮」而非「2N 条消息」计数。
    """
    if rounds <= 0:
        return list(messages)
    human_idx = [i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"]
    if len(human_idx) <= rounds:
        return list(messages)
    return list(messages[human_idx[len(human_idx) - rounds] :])


def _estimate_tokens(text: str) -> int:
    """无第三方 tokenizer 时的保守估算：CJK 约 1 字/token，其余约 4 字/token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    return cjk + (len(text) - cjk + 3) // 4


def _build_system(base: str, summary: str, memories: list[str]) -> str:
    """把 persona 基座与（可选的）摘要、召回记忆拼成 system 文本。"""
    parts = [base]
    if summary:
        parts.append(f"【本会话摘要】\n{summary}")
    if memories:
        bullet = "\n".join(f"- {m}" for m in memories)
        parts.append(f"【你记得的相关信息】\n{bullet}")
    return "\n\n".join(p for p in parts if p)


async def compose_prompt(
    state: RolePlayState, *, token_budget: int = 8000, recent_rounds: int = 10
) -> dict:
    base = state.get("system_prompt", "")
    memories = list(state.get("recalled_memories") or [])
    summary = state.get("summary", "") or ""

    # 自主闲聊：临时把"主动开口"指引拼到 system 末尾（不改人设基座，不影响正常轮）
    idle_scope = state.get("idle_scope") or ""
    if idle_scope:
        base = f"{base}\n\n{build_idle_instruction(idle_scope)}"

    recent = _to_chat_messages(_recent_rounds(state.get("messages", []), recent_rounds))

    system = _build_system(base, summary, memories)

    def _total() -> int:
        return _estimate_tokens(system) + sum(_estimate_tokens(m["content"]) for m in recent)

    # 按优先级丢弃：先摘要，再召回记忆，最后从最旧的轮次开始裁近 N 轮窗口
    if _total() > token_budget and summary:
        system = _build_system(base, "", memories)
    if _total() > token_budget and memories:
        system = _build_system(base, "", [])
    while _total() > token_budget and len(recent) > 1:
        recent.pop(0)

    prompt = [{"role": "system", "content": system}]
    prompt.extend(recent)
    # 闲聊轮：透传给 LLM 的消息若以 assistant 回合收尾，模型易退化为「只输出内心独白、无正文」。
    # 末尾补一条合成 user 提示（不进状态/记忆/快照），使 prompt 以 user 收尾即可纠正。
    if idle_scope:
        prompt.append({"role": "user", "content": build_idle_nudge(idle_scope)})
    return {"prompt": prompt}
