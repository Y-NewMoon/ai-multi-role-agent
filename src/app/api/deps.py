"""API 依赖：从 app.state 取服务实例（lifespan 中装配）。"""

from fastapi import Request

from app.lore.service import LoreService
from app.services.agent_service import AgentService
from app.services.chat_service import ChatService
from app.services.group_service import GroupService
from app.services.memory_api_service import MemoryApiService
from app.services.reminder_service import ReminderService
from app.services.session_service import SessionService
from app.skills.registry import SkillRegistry


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_group_service(request: Request) -> GroupService:
    return request.app.state.group_service


def get_lore_service(request: Request) -> LoreService:
    return request.app.state.lore_service


def get_reminder_service(request: Request) -> ReminderService:
    return request.app.state.reminder_service


def get_memory_api_service(request: Request) -> MemoryApiService:
    """V3 M2：本机记忆接口服务（lifespan 装配）。"""
    return request.app.state.memory_api_service


def get_skill_registry(request: Request) -> SkillRegistry | None:
    """工具注册表（V2.2）；未装配时返回 None（测试替身应用）。"""
    return getattr(request.app.state, "skill_registry", None)
