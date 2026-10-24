"""Skill 注册表（契约 §2.2）。"""

import logging

from app.skills.base import Skill

logger = logging.getLogger(__name__)


def _is_available(skill: Skill) -> bool:
    """兼容没有 `is_available` 的自定义 Skill（V2.4 B-01：不强制改 Protocol）。"""
    probe = getattr(skill, "is_available", None)
    return True if not callable(probe) else bool(probe())


class SkillRegistry:
    """工具注册表：重复注册直接失败（启动期），未知名字在 resolve 时忽略并告警。"""

    def __init__(self, skills: list[Skill]) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise ValueError(f"重复注册的工具名：{skill.name}")
            self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def resolve(self, allowed: list[str], *, only_available: bool = False) -> list[Skill]:
        """按白名单过滤；未知名字忽略并记 warning，不抛异常。

        `only_available=True`（V2.4 B-01）时再过滤 `is_available()` 为假的工具；
        对没有该方法的自定义 Skill 视为可用（兼容 V2.2 实现）。
        """
        out: list[Skill] = []
        for name in allowed:
            skill = self._skills.get(name)
            if skill is None:
                logger.warning("忽略未注册的工具名：%s", name)
                continue
            if only_available and not _is_available(skill):
                continue
            out.append(skill)
        return out

    def to_openai_tools(self, allowed: list[str], *, only_available: bool = True) -> list[dict]:
        """OpenAI tools schema；默认不下发不可用工具（V2.4 B-01）。"""
        return [
            skill.to_openai_tool()
            for skill in self.resolve(allowed, only_available=only_available)
        ]

    def names(self) -> list[str]:
        return list(self._skills)
