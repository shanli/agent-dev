"""LLM Gateway：统一门面，编排所有横切能力。

职责：
1. Prompt 解析与渲染
2. 按模型限流（超限 429）
3. 指数退避重试（max_retries=2，共 3 次尝试）
4. 适配器路由
5. 结构化输出校验 + 修复
6. 全链路可观测（token/延时/成本/错误）

流式关键约束：首 token 之前（WAITING）可安全重试；
首 token 之后（STREAMING）任何错误只发 error chunk，不静默重试。
"""

from __future__ import annotations

from typing import Optional

import asyncio
import time
from collections.abc import AsyncIterator

import httpx

from app.adapters.base import AdapterResult, NormalizedRequest
from app.adapters.registry import AdapterRegistry
from app.core.errors import LLMError, LLMErrorCode, LLMStage
from app.core.types import UnifiedChunk, UnifiedMessage, UnifiedRequest, UnifiedResponse, Usage
from app.gateway.prompt.manager import PromptManager
from app.gateway.resilience.ratelimit import ModelRateLimiter
from app.gateway.resilience.retry import with_retry
from app.gateway.structured.validator import StructuredValidator
from app.observability.telemetry import CallRecord, Telemetry


class LLMGateway:
    def __init__(
        self,
        registry: AdapterRegistry,
        prompt_manager: PromptManager,
        limiter: ModelRateLimiter,
        telemetry: Telemetry,
        validator: StructuredValidator,
    ) -> None:
        self._registry = registry
        self._prompt = prompt_manager
        self._limiter = limiter
        self._telemetry = telemetry
        self._validator = validator

    # ── 非流式 ──
    async def chat(self, req: UnifiedRequest) -> UnifiedResponse:
        rec = self._telemetry.start(req)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + req.timeout_ms / 1000

        try:
            messages = await self._resolve_messages(req, rec)
            norm = self._normalize(req, messages)
            est_tokens = self._estimate_tokens(messages, req.max_tokens)
            rec.provider = self._registry.get_config(req.model).provider

            async def call(attempt: int) -> UnifiedResponse:
                rec.attempt = attempt
                t0 = time.perf_counter()
                await self._limiter.acquire(req.model, est_tokens)
                rec.queue_ms = int((time.perf_counter() - t0) * 1000)
                result = await self._invoke_adapter(req, norm)
                return self._to_response(result, rec, attempt)

            resp = await with_retry(call, deadline=deadline, loop_time=loop.time)

            # 结构化输出校验 + 1 次修复（独立预算）
            if req.response_schema:
                resp = await self._validator.validate_and_repair(
                    resp, req, norm, rec, self._invoke_for_repair
                )

            await self._telemetry.finalize(rec, resp)
            return resp
        except LLMError as e:
            await self._telemetry.fail(rec, e)
            raise

    # ── 流式 ──
    async def chat_stream(self, req: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        rec = self._telemetry.start(req)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + req.timeout_ms / 1000

        try:
            messages = await self._resolve_messages(req, rec)
            norm = self._normalize(req, messages)
            est_tokens = self._estimate_tokens(messages, req.max_tokens)
            rec.provider = self._registry.get_config(req.model).provider
            timeout = httpx.Timeout(req.timeout_ms / 1000)

            async def open_stream(attempt: int):
                rec.attempt = attempt
                await self._limiter.acquire(req.model, est_tokens)
                adapter = self._registry.resolve(req.model)
                gen = adapter.chat_stream(norm, timeout)
                # 实际取首 chunk：确认连接成功 + 上游无错误（首 token 前可安全重试）
                first = await gen.__anext__()
                return first, gen

            # 仅 WAITING 期（首 chunk 之前）可重试，含认证失败/网络错误/上游 429
            first_chunk, adapter_iter = await with_retry(open_stream, deadline=deadline, loop_time=loop.time)

            # 首 chunk 成功 → 进入 STREAMING，后续不再重试
            state = "STREAMING"
            buf = ""
            usage = Usage()
            finish_reason = "stop"
            first_ts = time.perf_counter()
            rec.ttft_ms = int((first_ts - rec.started_perf) * 1000)

            # 处理首 chunk
            if first_chunk.type == "text_delta":
                buf += first_chunk.delta or ""
                yield UnifiedChunk(type="text_delta", delta=first_chunk.delta)
            elif first_chunk.type == "usage":
                usage = first_chunk.usage or usage
                yield UnifiedChunk(type="usage", usage=first_chunk.usage)
            elif first_chunk.type == "done":
                finish_reason = first_chunk.finish_reason or "stop"
                yield UnifiedChunk(type="done", finish_reason=finish_reason)
            elif first_chunk.type == "error" and first_chunk.error:
                raise first_chunk.error

            try:
                async for chunk in adapter_iter:
                    if chunk.type == "text_delta":
                        buf += chunk.delta or ""
                        yield UnifiedChunk(type="text_delta", delta=chunk.delta)
                    elif chunk.type == "usage":
                        usage = chunk.usage or usage
                        yield UnifiedChunk(type="usage", usage=chunk.usage)
                    elif chunk.type == "done":
                        finish_reason = chunk.finish_reason or "stop"
                        yield UnifiedChunk(type="done", finish_reason=finish_reason)
                    elif chunk.type == "error" and chunk.error:
                        raise chunk.error

                rec.input_tokens = usage.input_tokens
                rec.output_tokens = usage.output_tokens
                rec.cached_tokens = usage.cached_tokens
                rec.reasoning_tokens = usage.reasoning_tokens
                rec.finish_reason = finish_reason
                rec.final_state = "success"
                rec.total_latency_ms = int((time.perf_counter() - rec.started_perf) * 1000)
                if first_ts is not None:
                    rec.generation_ms = rec.total_latency_ms - rec.ttft_ms - (rec.queue_ms or 0)
                rec.is_final = True
                # 成本估算（复用 telemetry 内部逻辑：直接计算并落库）
                await self._finalize_stream(rec, usage)
            except LLMError as e:
                # mid_stream：已有部分输出交付，禁止盲目重新生成，发 error chunk 收尾
                rec.error_code = e.code.value
                rec.http_status = e.http_status
                rec.provider_request_id = e.provider_request_id
                rec.final_state = "error"
                rec.total_latency_ms = int((time.perf_counter() - rec.started_perf) * 1000)
                rec.is_final = True
                await self._telemetry.fail(rec, e)
                yield UnifiedChunk(
                    type="error",
                    error_code=e.code.value,
                    partial_content=buf,
                )
        except LLMError as e:
            # WAITING 期错误（Prompt/限流/路由/连接）：直接上抛，对外零感知
            await self._telemetry.fail(rec, e)
            raise

    # ── 内部方法 ──
    async def _resolve_messages(self, req: UnifiedRequest, rec: CallRecord) -> list[UnifiedMessage]:
        if req.prompt_ref:
            return await self._prompt.resolve(req.prompt_ref, rec)
        if req.messages:
            return req.messages
        raise LLMError(
            LLMErrorCode.INVALID_REQUEST,
            "必须提供 prompt_ref 或 messages",
            stage=LLMStage.VALIDATION,
        )

    def _normalize(self, req: UnifiedRequest, messages: list[UnifiedMessage]) -> NormalizedRequest:
        cfg = self._registry.get_config(req.model)
        return NormalizedRequest(
            actual_model=cfg.actual_model,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            response_schema=req.response_schema,
        )

    def _estimate_tokens(self, messages: list[UnifiedMessage], max_tokens: Optional[int]) -> int:
        """粗略估算：中文 1 字≈1 token，英文 4 字符≈1 token。"""
        total = 0
        for m in messages:
            for ch in m.content:
                total += 1 if ord(ch) > 127 else 0.25
        return int(total) + (max_tokens or 256)

    async def _invoke_adapter(self, req: UnifiedRequest, norm: NormalizedRequest) -> AdapterResult:
        adapter = self._registry.resolve(req.model)
        timeout = httpx.Timeout(req.timeout_ms / 1000)
        return await adapter.chat(norm, timeout)

    async def _invoke_for_repair(
        self, norm: NormalizedRequest, req: UnifiedRequest, rec: CallRecord, attempt: int
    ) -> UnifiedResponse:
        """供结构化修复复用的调用封装：限流 + 适配器调用 + 响应归一。"""
        await self._limiter.acquire(req.model, self._estimate_tokens(norm.messages, req.max_tokens))
        result = await self._invoke_adapter(req, norm)
        return self._to_response(result, rec, attempt)

    def _to_response(self, result: AdapterResult, rec: CallRecord, attempt: int) -> UnifiedResponse:
        total_latency_ms = int((time.perf_counter() - rec.started_perf) * 1000)
        rec.actual_model = result.actual_model
        rec.provider_request_id = result.request_id
        return UnifiedResponse(
            id=result.request_id or rec.call_id,
            logical_model=rec.logical_model,
            actual_model=result.actual_model,
            content=result.content,
            usage=result.usage,
            finish_reason=result.finish_reason,
            total_latency_ms=total_latency_ms,
            attempt=attempt,
            provider_request_id=result.request_id,
        )

    async def _finalize_stream(self, rec: CallRecord, usage: Usage) -> None:
        """流式成功收尾：直接复用 telemetry 的持久化与指标。"""
        # 计算成本
        cfg = self._registry.get_config(rec.logical_model) if hasattr(self._registry, "get_config") else None
        if cfg:
            price = cfg.price_per_million
            from decimal import Decimal

            rec.cost_input = Decimal(rec.input_tokens) / Decimal(1_000_000) * Decimal(price.input)
            rec.cost_output = Decimal(rec.output_tokens) / Decimal(1_000_000) * Decimal(price.output)
            rec.cost_cached = Decimal(rec.cached_tokens) / Decimal(1_000_000) * Decimal(price.cached)
            rec.cost_total = (rec.cost_input or 0) + (rec.cost_output or 0) + (rec.cost_cached or 0)
        await self._telemetry._persist(rec)
        self._telemetry._emit_metrics(rec, status="success")
