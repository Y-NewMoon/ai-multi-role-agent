"""角色接口：CRUD（M3）+ `allowed_skills` 工具白名单（V2.2 §2.8）。"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.deps import get_agent_service
from app.schemas.agent import AgentCreate, AgentRead, AgentUpdate
from app.services.agent_service import AgentService
from app.skills.registry import SkillRegistry

router = APIRouter(prefix="/agents", tags=["agents"])


def _validate_allowed_skills(request: Request, allowed: list[str] | None) -> None:
    """契约 §2.8：`allowed_skills` 校验用 registry；未知工具名 → 422；空数组合法。

    registry 由 lifespan 装配在 `app.state`；测试替身应用可缺省（此时不做校验）。
    """
    if not allowed:
        return
    registry: SkillRegistry | None = getattr(request.app.state, "skill_registry", None)
    if registry is None:
        return
    unknown = [name for name in allowed if registry.get(name) is None]
    if unknown:
        raise HTTPException(status_code=422, detail=f"未注册的工具名：{unknown}")


@router.get("", response_model=list[AgentRead])
async def list_agents(service: AgentService = Depends(get_agent_service)) -> list[AgentRead]:
    return await service.list()


@router.post("", response_model=AgentRead, status_code=status.HTTP_201_CREATED)
async def create_agent(
    payload: AgentCreate, request: Request, service: AgentService = Depends(get_agent_service)
) -> AgentRead:
    _validate_allowed_skills(request, payload.allowed_skills)
    return await service.create(payload)


@router.get("/{agent_id}", response_model=AgentRead)
async def get_agent(
    agent_id: int, service: AgentService = Depends(get_agent_service)
) -> AgentRead:
    return await service.get(agent_id)


@router.patch("/{agent_id}", response_model=AgentRead)
async def update_agent(
    agent_id: int,
    payload: AgentUpdate,
    request: Request,
    service: AgentService = Depends(get_agent_service),
) -> AgentRead:
    _validate_allowed_skills(request, payload.allowed_skills)
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
