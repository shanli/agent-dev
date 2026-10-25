"""适配器抽象基类与协议间通用数据类型。"""

from __future__ import annotations

from typing import Any, Optional

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel

from app.core.types import UnifiedMessage, Usage


class NormalizedRequest(BaseModel):
    """适配器接收的标准化请求（已由 Gateway 完成 Prompt 渲染）。"""

    actual_model: str
    messages: list[UnifiedMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_schema: Optional[dict] = None  # None = 普通对话
    schema_name: str = "structured_output"


class AdapterResult(BaseModel):
    """非流式调用结果。"""

    request_id: Optional[str]
    actual_model: str
    content: str
    usage: Usage
    finish_reason: str  # stop / length / content_filter
    raw_http_status: int


class AdapterChunk(BaseModel):
    """流式 chunk。适配器负责把协议特定事件归一成这三种类型之一。"""

    type: str  # text_delta / usage / done / error
    delta: Optional[str] = None
    usage: Optional[Usage] = None
    finish_reason: Optional[str] = None
    error: Optional[Any] = None  # type=error 时携带 LLMError 实例


AdapterChunk.model_rebuild()


class LLMAdapter(ABC):
    """适配器抽象基类。子类必须实现 chat 与 chat_stream。"""

    provider: str

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # 每个适配器一个 client，连接池复用
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(60.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    @abstractmethod
    async def chat(self, req: NormalizedRequest, timeout: httpx.Timeout) -> AdapterResult:
        """非流式调用。"""

    @abstractmethod
    def chat_stream(self, req: NormalizedRequest, timeout: httpx.Timeout) -> AsyncIterator[AdapterChunk]:
        """流式调用，返回异步迭代器。"""
