"""阶段化错误模型。

核心设计：
- 每个错误归属一个管线阶段（LLMStage），对应'错误处理、重试、限流与资源边界'图。
- 错误码 → (retryable, fallback_allowed) 映射表，让'内容拒答不换模型绕过'
  成为类型系统保证，而非开发自觉。
- RetryExecutor 只认 retryable；路由 fallback 只认 fallback_allowed。
"""

from __future__ import annotations

from typing import Optional

from enum import Enum


class LLMStage(str, Enum):
    """管线阶段。错误按阶段分类，不同阶段重试语义不同。"""

    AUTH = "auth"
    VALIDATION = "validation"
    PROMPT = "prompt"
    ROUTING = "routing"
    QUEUE = "queue"
    CONNECTION = "connection"
    PRE_FIRST_TOKEN = "pre_first_token"
    MID_STREAM = "mid_stream"
    OUTPUT_VALIDATION = "output_validation"
    CONTENT_REFUSAL = "content_refusal"


class LLMErrorCode(str, Enum):
    """稳定错误码，对外暴露并落库存储。"""

    # 不可重试
    AUTH_FAILED = "AUTH_FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"
    PROMPT_VARIABLE_MISSING = "PROMPT_VARIABLE_MISSING"
    CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"
    NO_COMPATIBLE_MODEL = "NO_COMPATIBLE_MODEL"
    CONTENT_FILTERED = "CONTENT_FILTERED"
    STREAM_INTERRUPTED = "STREAM_INTERRUPTED"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    UNKNOWN = "UNKNOWN"
    # 本地限流：不重试，但允许切换到候选池（fallback）
    RATE_LIMITED = "RATE_LIMITED"
    # 上游类：有限重试
    UPSTREAM_RATE_LIMITED = "UPSTREAM_RATE_LIMITED"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    UPSTREAM_OVERLOADED = "UPSTREAM_OVERLOADED"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    NETWORK_ERROR = "NETWORK_ERROR"


# 错误码 → (retryable, fallback_allowed)
# 对齐阶段化决策矩阵：
# - 认证/请求校验/Prompt 缺失/上下文超预算/无兼容模型：既不重试也不 fallback
# - 本地限流 RATE_LIMITED：不重试，但可转投候选池
# - 网络/5xx/上游超时/上游 429/过载：可重试；其中首 Token 前的上游类错误允许 fallback
# - 流式中断：禁止盲目重新生成
# - 输出校验失败：走独立修复预算，不走网络重试
# - 内容拒答：红线，禁止换模型绕过
RETRY_POLICY: dict[LLMErrorCode, tuple[bool, bool]] = {
    LLMErrorCode.AUTH_FAILED: (False, False),
    LLMErrorCode.INVALID_REQUEST: (False, False),
    LLMErrorCode.PROMPT_VARIABLE_MISSING: (False, False),
    LLMErrorCode.CONTEXT_BUDGET_EXCEEDED: (False, False),
    LLMErrorCode.NO_COMPATIBLE_MODEL: (False, False),
    LLMErrorCode.CONTENT_FILTERED: (False, False),
    LLMErrorCode.STREAM_INTERRUPTED: (False, False),
    LLMErrorCode.SCHEMA_VALIDATION_FAILED: (False, False),
    LLMErrorCode.DEADLINE_EXCEEDED: (False, False),
    LLMErrorCode.UNKNOWN: (False, False),
    LLMErrorCode.RATE_LIMITED: (False, True),
    LLMErrorCode.NETWORK_ERROR: (True, False),
    LLMErrorCode.UPSTREAM_UNAVAILABLE: (True, False),
    LLMErrorCode.UPSTREAM_TIMEOUT: (True, True),
    LLMErrorCode.UPSTREAM_RATE_LIMITED: (True, True),
    LLMErrorCode.UPSTREAM_OVERLOADED: (True, True),
}


class LLMError(Exception):
    """统一异常。所有适配器抛出的异常最终都归一为此类型。"""

    def __init__(
        self,
        code: LLMErrorCode,
        message: str,
        *,
        stage: LLMStage,
        http_status: Optional[int] = None,
        provider_request_id: Optional[str] = None,
        retry_after_ms: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.http_status = http_status
        self.provider_request_id = provider_request_id
        self.retry_after_ms = retry_after_ms
        self.cause = cause
        self.retryable, self.fallback_allowed = RETRY_POLICY[code]

    def to_dict(self) -> dict:
        d = {"code": self.code.value, "stage": self.stage.value, "message": self.message}
        if self.http_status is not None:
            d["http_status"] = self.http_status
        if self.provider_request_id:
            d["provider_request_id"] = self.provider_request_id
        if self.retry_after_ms is not None:
            d["retry_after_ms"] = self.retry_after_ms
        return d
