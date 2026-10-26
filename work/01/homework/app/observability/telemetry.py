"""可观测性：分段计时 + 记录落库 + Prometheus 指标。

字段对齐'八、Token、Cost、Latency 与 Trace'九维。
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from prometheus_client import Counter, Histogram

from app.core.config import ModelConfig
from app.core.errors import LLMError
from app.core.types import UnifiedRequest, UnifiedResponse, Usage
from app.db.models import LLMCallRecord

logger = logging.getLogger(__name__)

# ── Prometheus 指标 ──
CALLS_TOTAL = Counter(
    "llm_calls_total",
    "LLM 调用总数",
    ["model", "provider", "status", "error_code"],
)
TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "LLM token 消耗总数",
    ["model", "direction"],  # direction: input/output/cached/reasoning
)
LATENCY_SECONDS = Histogram(
    "llm_latency_seconds",
    "LLM 调用各阶段延时",
    ["model", "phase"],  # phase: queue/ttft/generation/total
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)
RETRY_TOTAL = Counter("llm_retries_total", "LLM 重试次数", ["model", "error_code"])


@dataclass
class CallRecord:
    """内存中的调用记录，finalize/fail 后落库。字段与 LLMCallRecord 一一对应。"""

    # 关联
    trace_id: str
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    call_id: str = ""
    attempt: int = 1
    is_final: bool = False

    # 路由
    logical_model: str = ""
    provider: str = ""
    actual_model: Optional[str] = None
    candidates: Any = None
    reject_reason: Any = None

    # Prompt
    prompt_name: Optional[str] = None
    prompt_version: Optional[int] = None
    prompt_hash: Optional[str] = None
    schema_version: Optional[str] = None

    # 请求特征
    stream: bool = False
    structured: bool = False

    # 用量
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    # 延迟 (ms)
    queue_ms: Optional[int] = None
    route_ms: Optional[int] = None
    ttft_ms: Optional[int] = None
    generation_ms: Optional[int] = None
    total_latency_ms: Optional[int] = None

    # 弹性
    fallback_from: Optional[str] = None
    timeout_budget_ms: Optional[int] = None

    # 结果
    final_state: Optional[str] = None
    finish_reason: Optional[str] = None
    validation_result: Optional[str] = None

    # 错误
    error_code: Optional[str] = None
    http_status: Optional[int] = None
    provider_request_id: Optional[str] = None

    # 成本（写入时估算）
    cost_input: Optional[Decimal] = None
    cost_output: Optional[Decimal] = None
    cost_cached: Optional[Decimal] = None
    cost_total: Optional[Decimal] = None
    currency: str = "USD"

    # 时间戳（perf_counter 用于算差值）
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_perf: float = field(default_factory=time.perf_counter)


class Telemetry:
    def __init__(self, session_factory: Any, model_configs: dict[str, ModelConfig]) -> None:
        self._session_factory = session_factory
        self._configs = model_configs

    def start(self, req: UnifiedRequest) -> CallRecord:
        """创建调用记录，生成 trace_id/call_id。"""
        rec = CallRecord(
            trace_id=req.trace_id or secrets.token_hex(8),
            run_id=req.run_id,
            step_id=req.step_id,
            call_id=secrets.token_hex(8),
            logical_model=req.model,
            stream=req.stream,
            structured=req.response_schema is not None,
            schema_version=req.schema_version,
            timeout_budget_ms=req.timeout_ms,
        )
        return rec

    async def finalize(self, rec: CallRecord, resp: UnifiedResponse) -> None:
        """成功收尾：填充结果字段，计算成本，落库 + 指标。"""
        elapsed = time.perf_counter() - rec.started_perf
        rec.actual_model = resp.actual_model
        rec.input_tokens = resp.usage.input_tokens
        rec.output_tokens = resp.usage.output_tokens
        rec.cached_tokens = resp.usage.cached_tokens
        rec.reasoning_tokens = resp.usage.reasoning_tokens
        rec.total_latency_ms = resp.total_latency_ms
        rec.ttft_ms = resp.ttft_ms
        rec.finish_reason = resp.finish_reason
        rec.provider_request_id = resp.provider_request_id
        rec.final_state = "success"
        rec.is_final = True

        # 分段延时（generation = total - ttft - queue）
        if rec.ttft_ms is not None and rec.total_latency_ms is not None:
            rec.generation_ms = max(
                0,
                rec.total_latency_ms - rec.ttft_ms - (rec.queue_ms or 0),
            )

        # 成本估算
        cfg = self._configs.get(rec.logical_model)
        if cfg:
            price = cfg.price_per_million
            rec.cost_input = Decimal(rec.input_tokens) / Decimal(1_000_000) * Decimal(price.input)
            rec.cost_output = Decimal(rec.output_tokens) / Decimal(1_000_000) * Decimal(price.output)
            rec.cost_cached = Decimal(rec.cached_tokens) / Decimal(1_000_000) * Decimal(price.cached)
            rec.cost_total = (rec.cost_input or 0) + (rec.cost_output or 0) + (rec.cost_cached or 0)

        await self._persist(rec)
        self._emit_metrics(rec, status="success")

    async def fail(self, rec: CallRecord, error: LLMError) -> None:
        """失败收尾：填充错误字段，落库 + 指标。"""
        rec.error_code = error.code.value
        rec.http_status = error.http_status
        rec.provider_request_id = error.provider_request_id
        rec.final_state = "error"
        rec.is_final = True
        rec.total_latency_ms = int((time.perf_counter() - rec.started_perf) * 1000)
        await self._persist(rec)
        self._emit_metrics(rec, status="error", error_code=error.code.value)
        logger.warning(
            "llm.call_failed trace_id=%s model=%s code=%s stage=%s",
            rec.trace_id,
            rec.logical_model,
            error.code,
            error.stage,
        )

    async def _persist(self, rec: CallRecord) -> None:
        """把内存记录写入数据库。"""
        try:
            async with self._session_factory() as session:
                row = LLMCallRecord(
                    trace_id=rec.trace_id,
                    run_id=rec.run_id,
                    step_id=rec.step_id,
                    call_id=rec.call_id,
                    attempt=rec.attempt,
                    is_final=rec.is_final,
                    logical_model=rec.logical_model,
                    provider=rec.provider,
                    actual_model=rec.actual_model,
                    candidates=rec.candidates,
                    reject_reason=rec.reject_reason,
                    prompt_name=rec.prompt_name,
                    prompt_version=rec.prompt_version,
                    prompt_hash=rec.prompt_hash,
                    schema_version=rec.schema_version,
                    stream=rec.stream,
                    structured=rec.structured,
                    input_tokens=rec.input_tokens,
                    output_tokens=rec.output_tokens,
                    cached_tokens=rec.cached_tokens,
                    reasoning_tokens=rec.reasoning_tokens,
                    queue_ms=rec.queue_ms,
                    route_ms=rec.route_ms,
                    ttft_ms=rec.ttft_ms,
                    generation_ms=rec.generation_ms,
                    total_latency_ms=rec.total_latency_ms,
                    fallback_from=rec.fallback_from,
                    timeout_budget_ms=rec.timeout_budget_ms,
                    final_state=rec.final_state,
                    finish_reason=rec.finish_reason,
                    validation_result=rec.validation_result,
                    error_code=rec.error_code,
                    http_status=rec.http_status,
                    provider_request_id=rec.provider_request_id,
                    cost_input=rec.cost_input,
                    cost_output=rec.cost_output,
                    cost_cached=rec.cost_cached,
                    cost_total=rec.cost_total,
                    started_at=rec.started_at,
                    finished_at=datetime.now(timezone.utc),
                )
                session.add(row)
                await session.commit()
        except Exception:
            logger.exception("telemetry persist failed trace_id=%s", rec.trace_id)

    @staticmethod
    def _emit_metrics(rec: CallRecord, *, status: str, error_code: str = "") -> None:
        model = rec.logical_model
        CALLS_TOTAL.labels(model=model, provider=rec.provider, status=status, error_code=error_code).inc()
        TOKENS_TOTAL.labels(model=model, direction="input").inc(rec.input_tokens)
        TOKENS_TOTAL.labels(model=model, direction="output").inc(rec.output_tokens)
        TOKENS_TOTAL.labels(model=model, direction="cached").inc(rec.cached_tokens)
        TOKENS_TOTAL.labels(model=model, direction="reasoning").inc(rec.reasoning_tokens)
        if rec.total_latency_ms is not None:
            LATENCY_SECONDS.labels(model=model, phase="total").observe(rec.total_latency_ms / 1000)
        if rec.ttft_ms is not None:
            LATENCY_SECONDS.labels(model=model, phase="ttft").observe(rec.ttft_ms / 1000)
        if rec.queue_ms is not None:
            LATENCY_SECONDS.labels(model=model, phase="queue").observe(rec.queue_ms / 1000)
        if rec.generation_ms is not None:
            LATENCY_SECONDS.labels(model=model, phase="generation").observe(rec.generation_ms / 1000)
