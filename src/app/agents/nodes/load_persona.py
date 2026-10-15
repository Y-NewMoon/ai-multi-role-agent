"""节点：装配 system prompt（人设注入）。

图内不访问业务表：人设来自状态里服务层预取的字段（接缝 5）。
"""

from app.agents.state import RolePlayState
from app.prompts.roleplay_system import build_system_prompt


async def load_persona(state: RolePlayState) -> dict:
    system_prompt = build_system_prompt(
        name=state.get("agent_name", ""),
        persona=state.get("persona_prompt", ""),
        background=state.get("background") or "",
    )
    return {"system_prompt": system_prompt}
