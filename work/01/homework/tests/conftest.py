"""测试基础设施：SQLite 内存库 + Gateway fixture + respx mock。"""

from __future__ import annotations

import os

# 在导入任何 app 模块前，强制使用 SQLite 内存库（避免触发 asyncpg 导入）
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("TEST_API_KEY", "sk-test-key")

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.adapters.registry import AdapterRegistry
from app.core.config import ModelConfig, PricePerMillion
from app.db.models import Base
from app.gateway.gateway import LLMGateway
from app.gateway.prompt.manager import PromptManager
from app.gateway.resilience.ratelimit import ModelRateLimiter
from app.gateway.structured.validator import StructuredValidator
from app.observability.telemetry import Telemetry

# ── 测试用模型配置（高限流，避免触发本地 429）──
TEST_MODELS: dict[str, ModelConfig] = {
    "deepseek-v4-pro": ModelConfig(
        key="deepseek-v4-pro",
        provider="openai_responses",
        actual_model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_key_env="TEST_API_KEY",
        rpm=100000,
        tpm=10000000,
        price_per_million=PricePerMillion(input=0.27, output=1.10, cached=0.07),
    ),
    "deepseek-v4-flash": ModelConfig(
        key="deepseek-v4-flash",
        provider="anthropic_messages",
        actual_model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/anthropic",
        api_key_env="TEST_API_KEY",
        rpm=100000,
        tpm=10000000,
        price_per_million=PricePerMillion(input=0.07, output=0.28, cached=0.02),
    ),
}


@pytest.fixture(scope="session")
def event_loop_policy():
    import asyncio

    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(db_engine):
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    yield factory


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncIterator[AsyncSession]:
    async with session_factory() as s:
        yield s


@pytest_asyncio.fixture
def registry():
    os.environ["TEST_API_KEY"] = "sk-test-key"
    return AdapterRegistry(TEST_MODELS)


@pytest_asyncio.fixture
def limiter():
    return ModelRateLimiter(TEST_MODELS)


@pytest_asyncio.fixture
def telemetry(session_factory):
    return Telemetry(session_factory, TEST_MODELS)


@pytest_asyncio.fixture
def prompt_manager(session_factory):
    return PromptManager(session_factory)


@pytest_asyncio.fixture
def validator():
    return StructuredValidator()


@pytest_asyncio.fixture
def gateway(registry, limiter, telemetry, prompt_manager, validator):
    return LLMGateway(
        registry=registry,
        prompt_manager=prompt_manager,
        limiter=limiter,
        telemetry=telemetry,
        validator=validator,
    )


@pytest.fixture
def sample_messages():
    from app.core.types import UnifiedMessage

    return [
        UnifiedMessage(role="system", content="你是助手"),
        UnifiedMessage(role="user", content="你好"),
    ]


@pytest_asyncio.fixture
async def app(gateway, session_factory, db_engine):
    """构建一个使用测试配置的 FastAPI 应用。"""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from app.api.routes import router
    from app.core.errors import LLMError, LLMErrorCode
    from app.db import session as db_session

    # 覆盖全局 session 模块的 engine，让路由依赖注入用测试库
    db_session.engine = db_engine
    db_session.AsyncSessionLocal = session_factory

    STATUS_MAP = {
        LLMErrorCode.RATE_LIMITED: 429,
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

    application = FastAPI()
    application.include_router(router)

    @application.exception_handler(LLMError)
    async def llm_error_handler(request, exc: LLMError):
        status = STATUS_MAP.get(exc.code, 502)
        return JSONResponse(status_code=status, content={"error": exc.to_dict()})

    application.state.gateway = gateway
    yield application


@pytest.fixture
def openai_success_response():
    return {
        "id": "resp_123",
        "model": "deepseek-v4-pro",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "你好！有什么可以帮你的？"}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


@pytest.fixture
def anthropic_success_response():
    return {
        "id": "msg_123",
        "model": "deepseek-v4-flash",
        "content": [{"type": "text", "text": "你好！"}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 1},
        "stop_reason": "end_turn",
    }
