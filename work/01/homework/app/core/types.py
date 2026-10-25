"""统一数据模型：跨协议的请求/响应/流式 chunk。

所有适配器必须把协议特定的响应归一化到这里定义的类型，
Gateway 只与统一类型打交道，从而屏蔽协议差异。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """统一用量模型，对齐可观测性'用量'维度：input/output/cached/reasoning。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UnifiedMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class PromptRef(BaseModel):
    """提示词引用：通过 name + version 取模板，variables 用于渲染。"""

    name: str
    version: Optional[int] = None  # None → 当前激活版本（is_active=True）
    variables: dict[str, Any] = Field(default_factory=dict)


class UnifiedRequest(BaseModel):
    """统一调用请求。Gateway 入口只接受此模型。"""

    model: str
    prompt_ref: Optional[PromptRef] = None
    messages: Optional[list[UnifiedMessage]] = None  # 与 prompt_ref 二选一
    response_schema: Optional[dict[str, Any]] = None  # 传入即启用结构化输出
    schema_version: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout_ms: int = 60_000  # 截止时间预算（含重试）
    trace_id: Optional[str] = None  # 可由上游透传，用于全链路关联
    run_id: Optional[str] = None
    step_id: Optional[str] = None


class UnifiedResponse(BaseModel):
    """统一非流式响应。"""

    id: str
    logical_model: str
    actual_model: str
    content: str  # 结构化输出时为已校验的 JSON 字符串
    usage: Usage
    finish_reason: Literal["stop", "length", "content_filter"]
    total_latency_ms: int
    ttft_ms: Optional[int] = None
    attempt: int = 1
    provider_request_id: Optional[str] = None


class UnifiedChunk(BaseModel):
    """统一流式 chunk。"""

    type: Literal["text_delta", "usage", "done", "error"]
    delta: Optional[str] = None
    usage: Optional[Usage] = None
    finish_reason: Optional[str] = None
    # type=error 时：mid_stream 阶段才会出现，附带已流出的部分内容
    error_code: Optional[str] = None
    partial_content: Optional[str] = None
