"""Agent CRUD 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from agent_api.db import InMemoryDB
from agent_api.dependencies import get_db, verify_admin
from agent_api.models import (
    AgentListResponse,
    AgentResponse,
    CreateAgentRequest,
    Pagination,
    UpdateAgentRequest,
)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
def create_agent(
    body: CreateAgentRequest,
    db: InMemoryDB = Depends(get_db),
    admin_key: str = Depends(verify_admin),
) -> AgentResponse:
    """创建 Agent。"""
    data = body.model_dump()
    agent = db.create(data)
    return agent


@router.get("", response_model=AgentListResponse)
def list_agents(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: InMemoryDB = Depends(get_db),
    admin_key: str = Depends(verify_admin),
) -> AgentListResponse:
    """分页查询 Agent 列表。"""
    all_agents = db.list_all()
    total = len(all_agents)
    start = (page - 1) * page_size
    end = start + page_size
    items = all_agents[start:end]
    return AgentListResponse(
        items=items,
        pagination=Pagination(page=page, page_size=page_size, total=total),
    )


@router.get("/search", response_model=list[AgentResponse])
def search_agents(
    q: str = Query(..., min_length=1, description="搜索关键词（按名称模糊匹配，大小写不敏感）"),
    db: InMemoryDB = Depends(get_db),
    admin_key: str = Depends(verify_admin),
) -> list[AgentResponse]:
    """按名称模糊搜索 Agent。"""
    return db.search(q)


@router.get("/{agent_id}", response_model=AgentResponse)
def get_agent(
    agent_id: str,
    db: InMemoryDB = Depends(get_db),
    admin_key: str = Depends(verify_admin),
) -> AgentResponse:
    """获取单个 Agent 详情。"""
    agent = db.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {agent_id} 不存在")
    return agent


@router.put("/{agent_id}", response_model=AgentResponse)
def update_agent(
    agent_id: str,
    body: UpdateAgentRequest,
    db: InMemoryDB = Depends(get_db),
    admin_key: str = Depends(verify_admin),
) -> AgentResponse:
    """更新 Agent（仅更新传入的字段）。"""
    data = body.model_dump(exclude_unset=True)
    agent = db.update(agent_id, data)
    if agent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {agent_id} 不存在")
    return agent


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent(
    agent_id: str,
    db: InMemoryDB = Depends(get_db),
    admin_key: str = Depends(verify_admin),
) -> None:
    """删除 Agent。"""
    ok = db.delete(agent_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {agent_id} 不存在")
