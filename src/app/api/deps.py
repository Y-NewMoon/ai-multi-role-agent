"""API 依赖：从 app.state 取服务实例（lifespan 中装配）。"""

from fastapi import Request

from app.services.agent_service import AgentService
from app.services.chat_service import ChatService
from app.services.group_service import GroupService
from app.services.session_service import SessionService


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_group_service(request: Request) -> GroupService:
    return request.app.state.group_service
