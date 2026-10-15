"""滚动摘要（方案 3.10）。

工作记忆超过 N 轮时，把最旧 M 轮压缩为 1 条摘要，写入该 agent 的 Store namespace，
key = f"summary:{session_id}"（由 MemoryStore.put_summary 落库，图内不碰业务表）。

摘要用「轻模型」做也可（方案 3.5/3.13：对话与摘要走同一 Provider，本项目共用 LLM）。
"""

from langchain_core.messages import BaseMessage

from app.llm.base import LLMProvider
from app.prompts.summarize import SUMMARIZE_PROMPT

_ROLE_LABEL = {"human": "用户", "ai": "角色", "system": "系统"}


def _format_conversation(messages: list[BaseMessage], existing_summary: str = "") -> str:
    """把待压缩的对话（含已有摘要）拼成纯文本，供摘要 prompt 使用。"""
    lines: list[str] = []
    if existing_summary.strip():
        lines.append(f"【此前摘要】{existing_summary.strip()}")
    for m in messages:
        label = _ROLE_LABEL.get(getattr(m, "type", "human"), "用户")
        content = str(m.content).strip()
        if content:
            lines.append(f"{label}：{content}")
    return "\n".join(lines)


async def summarize(
    llm: LLMProvider,
    messages: list[BaseMessage],
    *,
    existing_summary: str = "",
) -> str:
    """把 `messages`（可叠加已有摘要）压缩成一段要点摘要文本。"""
    conversation = _format_conversation(messages, existing_summary)
    if not conversation:
        return ""
    prompt = [
        {"role": "system", "content": "你是对话摘要助手，只输出摘要正文。"},
        {"role": "user", "content": SUMMARIZE_PROMPT.format(conversation=conversation)},
    ]
    return (await llm.generate(prompt)).strip()
