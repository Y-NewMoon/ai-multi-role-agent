"""v1 路由聚合。"""

from fastapi import APIRouter

from app.api.v1 import agents, chat, group, sessions, uploads

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(chat.router)
api_router.include_router(agents.router)
api_router.include_router(sessions.router)
api_router.include_router(group.router)
api_router.include_router(uploads.router)
