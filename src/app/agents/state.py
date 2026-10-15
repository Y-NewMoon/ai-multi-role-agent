"""图状态定义。

接缝 5（方案 3.11A#5）：persona / background 由服务层预取后注入本状态，
图内节点只读状态，不访问业务表。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RolePlayState(TypedDict, total=False):
    # —— 会话工作记忆（checkpointer 为真相源，add_messages 累积）——
    messages: Annotated[list[AnyMessage], add_messages]

    # —— 服务层预取的档案（图内只读）——
    agent_id: int
    session_id: int | str  # 群聊传入 f"g{group_session_id}"，用于摘要 key 前缀（ADR-0004 §3.6）
    agent_name: str
    persona_prompt: str
    background: str

    # —— 图内中间产物 ——
    system_prompt: str
    recalled_memories: list[str]
    summary: str  # 本会话滚动摘要（方案 3.10，由 retrieve_memory 取出）
    prompt: list[dict]
    reply: str
    # M5：本轮角色连发的消息条（1~N 条，已去分隔符）；1 条 = 1 条 AIMessage = 1 行快照
    replies: list[str]
    # M6：本轮每条发言对应的「内心独白」（思考链），与 replies 按 index 一一对齐；
    # 不含思考的发言为空串。仅作快照附加列由服务层落库，不进 messages/记忆。
    thinkings: list[str]

    # —— 自主闲聊标记（非空表示本次为角色自主闲聊）——
    # "private"：私聊闲聊；"group"：群聊闲聊。空/缺省 = 用户触发的正常回复。
    idle_scope: str
    # 本轮生成的消息条数上限覆盖值；0/缺省 = 用图构造时的默认（reply_max_messages）。
    reply_max_messages: int
