"""记忆隔离：namespace 只由 agent_id 决定。

接缝 2（方案 3.11A#2）：namespace 纯度——不掺 session / 群聊上下文，
保证同一角色在私聊与群聊中共享同一份记忆。
"""

MEMORY_SEGMENT = "memories"


def memory_namespace(agent_id: int | str) -> tuple[str, str]:
    """返回该角色的记忆 namespace。"""
    return (f"agent:{agent_id}", MEMORY_SEGMENT)
