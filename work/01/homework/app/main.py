"""FastAPI 应用入口：装配依赖、全局异常处理、metrics 暴露。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.adapters.registry import AdapterRegistry
from app.api.routes import router
from app.core.config import settings
from app.core.errors import LLMError, LLMErrorCode
from app.db.models import Base
from app.db.session import AsyncSessionLocal, engine
from app.gateway.gateway import LLMGateway
from app.gateway.prompt.manager import PromptManager
from app.gateway.prompt.repo import PromptRepo
from app.gateway.resilience.ratelimit import ModelRateLimiter
from app.gateway.structured.validator import StructuredValidator
from app.observability.telemetry import Telemetry

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── 错误码 → HTTP 状态映射 ──
STATUS_MAP: dict[LLMErrorCode, int] = {
    LLMErrorCode.RATE_LIMITED: 429,  # 需求：本地超限返回 429
    LLMErrorCode.AUTH_FAILED: 401,
    LLMErrorCode.INVALID_REQUEST: 400,
    LLMErrorCode.PROMPT_VARIABLE_MISSING: 422,
    LLMErrorCode.CONTEXT_BUDGET_EXCEEDED: 422,
    LLMErrorCode.NO_COMPATIBLE_MODEL: 404,
    LLMErrorCode.CONTENT_FILTERED: 422,
    LLMErrorCode.SCHEMA_VALIDATION_FAILED: 502,
    LLMErrorCode.UPSTREAM_TIMEOUT: 504,
    LLMErrorCode.DEADLINE_EXCEEDED: 504,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 启动：建表 + 装配依赖 ──
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    model_configs = settings.load_models()
    registry = AdapterRegistry(model_configs)
    limiter = ModelRateLimiter(model_configs)
    telemetry = Telemetry(AsyncSessionLocal, model_configs)
    validator = StructuredValidator()
    prompt_manager = PromptManager(AsyncSessionLocal)

    gateway = LLMGateway(
        registry=registry,
        prompt_manager=prompt_manager,
        limiter=limiter,
        telemetry=telemetry,
        validator=validator,
    )

    app.state.gateway = gateway
    app.state.prompt_manager = prompt_manager
    app.state.registry = registry
    app.state.model_configs = model_configs

    logger.info("LLM 统一调用服务启动，已注册模型: %s", list(model_configs.keys()))

    yield

    # ── 关闭 ──
    await registry.aclose()
    await engine.dispose()
    logger.info("LLM 统一调用服务已关闭")


app = FastAPI(title="LLM 统一调用服务", version="0.1.0", lifespan=lifespan)
app.include_router(router)


# ── 全局异常处理：LLMError → 统一错误响应 ──
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    status = STATUS_MAP.get(exc.code, 502)
    body = {"error": exc.to_dict()}
    headers = {}
    if exc.retry_after_ms:
        headers["Retry-After"] = str(max(1, round(exc.retry_after_ms / 1000)))
    return JSONResponse(status_code=status, content=body, headers=headers)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
