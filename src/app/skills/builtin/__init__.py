"""V2.2 内置工具（契约 §2.5）。"""

from app.skills.registry import SkillRegistry
from app.skills.builtin.dice import make_roll_dice_skill
from app.skills.builtin.read_lore import make_read_lore_skill
from app.skills.builtin.reminder import make_set_reminder_skill
from app.skills.builtin.time import get_current_time_skill
from app.skills.builtin.web_search import make_web_search_skill

__all__ = ["build_default_registry"]


def build_default_registry(
    *,
    search_provider=None,  # noqa: ANN001
    session_factory=None,  # noqa: ANN001
    lore_retriever=None,  # noqa: ANN001
) -> SkillRegistry:
    """组装 5 个内置工具；依赖（搜索 / DB / 检索器）由 main.py 注入。"""
    return SkillRegistry(
        [
            get_current_time_skill,
            make_roll_dice_skill(),
            make_web_search_skill(search_provider),
            make_set_reminder_skill(session_factory),
            make_read_lore_skill(lore_retriever),
        ]
    )
