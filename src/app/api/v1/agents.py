"""角色接口：CRUD（M3）。"""

from fastapi import APIRouter, Depends, Response, status

from app.api.deps import get_agent_service
from app.schemas.agent import AgentCreate, AgentRead, AgentUpdate
from app.services.agent_service import AgentService

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentRead])
async def list_agents(service: AgentService = Depends(get_agent_service)) -> list[AgentRead]:
    return await service.list()


@router.post("", response_model=AgentRead, status_code=status.HTTP_201_CREATED)
async def create_agent(
    payload: AgentCreate, service: AgentService = Depends(get_agent_service)
) -> AgentRead:
    return await service.create(payload)


@router.get("/{agent_id}", response_model=AgentRead)
async def get_agent(
    agent_id: int, service: AgentService = Depends(get_agent_service)
) -> AgentRead:
    return await service.get(agent_id)


@router.patch("/{agent_id}", response_model=AgentRead)
async def update_agent(
    agent_id: int, payload: AgentUpdate, service: AgentService = Depends(get_agent_service)
) -> AgentRead:
    return await service.update(agent_id, payload)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: int, service: AgentService = Depends(get_agent_service)
) -> Response:
    await service.delete(agent_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{agent_id}/memory/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_agent_memory(
    agent_id: int, service: AgentService = Depends(get_agent_service)
) -> Response:
    """清除该角色的记忆：长期记忆 + 全部对话历史（角色档案与会话壳保留）。"""
    await service.clear_memory(agent_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
