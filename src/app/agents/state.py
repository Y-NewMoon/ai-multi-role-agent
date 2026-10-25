"""图状态定义。

接缝 5（方案 3.11A#5）：persona / background 由服务层预取后注入本状态，
图内节点只读状态，不访问业务表。
"""

from typing import Annotated, TypedDict

import operator

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
    # V3 M1：上一条 user 消息的时间（ISO 8601 带偏移）；无历史时为空串/缺省。
    # 语义是“用户上次开口”，供 compose_prompt 渲染“你们上次说话是 …”。
    last_user_at: str

    # —— 图内中间产物 ——
    system_prompt: str
    recalled_memories: list[str]
    summary: str  # 本会话滚动摘要（方案 3.10，由 retrieve_memory 取出）
    # V2.3：记忆升级三段（默认空；extract 关闭时保持空 → prompt 与 V2.2 逐字一致）
    recalled_facts: list[dict]
    user_profile: dict[str, str]
    l2_portrait: str
    # V2.1：本轮注入的设定资料（世界书 chunk），元素
    # {"source_ref","header_path","content","score","routes"}；由 retrieve_lore 填充。
    recalled_lore: list[dict]
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

    # —— V2.2 Skill 工具层 ——
    # 本角色白名单（服务层从 Agent.allowed_skills 写入）；空 = 不绑工具。
    allowed_skills: list[str]
    # 已执行的工具轮数（每进一次 tools 节点 +1）。
    tool_rounds: int
    # 本轮模型请求的工具调用（generate 写入；tools 节点消费后清空）。
    tool_calls: list[dict]
    # 累计工具结果（结构化，operator.add 累加；供报告/测试断言）。
    # 注意：checkpointer 会跨轮保留，故每条记录带 `tool_turn` 标记，本轮只认同一标记的记录。
    tool_results: Annotated[list[dict], operator.add]
    # 本轮工具作用域标记（服务层每次请求显式重置，用于隔离上一轮遗留的 tool_results）。
    tool_turn: str

    # —— V2.4：本轮 LLM 走超时/降级兜底（降级轮不写记忆，§2.9）——
    degraded: bool
