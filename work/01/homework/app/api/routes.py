"""HTTP 路由：统一调用接口 + 提示词版本管理。"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.core.errors import LLMError, LLMErrorCode
from app.core.types import UnifiedRequest, UnifiedResponse
from app.gateway.prompt.manager import PromptManager
from app.gateway.prompt.repo import PromptRepo
from app.db.session import get_session

router = APIRouter()


def get_gateway(request: Request):
    return request.app.state.gateway


def get_prompt_manager(request: Request) -> PromptManager:
    return request.app.state.prompt_manager


# ── 统一调用 ──
@router.post("/v1/chat", response_model=UnifiedResponse)
async def chat(req: UnifiedRequest, gateway=Depends(get_gateway)):
    return await gateway.chat(req)


@router.post("/v1/chat/stream")
async def chat_stream(req: UnifiedRequest, gateway=Depends(get_gateway)):
    # 先取首个 chunk：WAITING 期错误（限流/认证/网络）在发响应头前就能确定 HTTP 状态码
    stream = gateway.chat_stream(req)
    try:
        first = await stream.__anext__()
    except LLMError as e:
        # 由全局异常处理器返回正确状态码
        raise
    except StopAsyncIteration:
        return StreamingResponse(iter([]), media_type="text/event-stream")

    async def gen():
        yield f"data: {first.model_dump_json()}\n\n"
        async for chunk in stream:
            yield f"data: {chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── 提示词版本管理 ──
class CreatePromptVersionReq(BaseModel):
    template: str
    variables: Optional[list[str]] = None


class ActivatePromptVersionReq(BaseModel):
    version: int


@router.post("/v1/prompts/{name}/versions")
async def create_prompt_version(
    name: str,
    body: CreatePromptVersionReq,
    session=Depends(get_session),
):
    repo = PromptRepo(session)
    pv = await repo.create_version(name, body.template, body.variables)
    return {"name": pv.name, "version": pv.version, "is_active": pv.is_active}


@router.post("/v1/prompts/{name}/versions/{version}/activate")
async def activate_prompt_version(name: str, version: int, session=Depends(get_session)):
    repo = PromptRepo(session)
    await repo.activate(name, version)
    return {"name": name, "active_version": version}


@router.get("/v1/prompts/{name}/versions")
async def list_prompt_versions(name: str, session=Depends(get_session)):
    repo = PromptRepo(session)
    versions = await repo.list_versions(name)
    return {
        "name": name,
        "versions": [
            {
                "version": v.version,
                "is_active": v.is_active,
                "variables": v.variables,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in versions
        ],
    }
