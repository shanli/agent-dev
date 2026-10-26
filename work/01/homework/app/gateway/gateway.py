"""LLM Gateway：统一门面，编排所有横切能力。

职责：
1. Prompt 解析与渲染
2. 按模型限流（超限 429）
3. 指数退避重试（max_retries=2，共 3 次尝试）
4. 适配器路由 + 候选池 fallback（本地限流/上游 429 时切换候选模型）
5. 结构化输出校验 + 修复
6. 全链路可观测（token/延时/成本/错误）

流式关键约束：首 token 之前（WAITING）可安全重试 + fallback；
首 token 之后（STREAMING）任何错误只发 error chunk，不静默重试、不切换模型。
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
            est_tokens = self._estimate_tokens(messages, req.max_tokens)

            # 构建模型链：主模型 + 候选池（去重、过滤未注册）
            model_chain = self._build_model_chain(req.model)

            last_error: Optional[LLMError] = None
            succeeded_model: Optional[str] = None
            norm: Optional[NormalizedRequest] = None

            for idx, model in enumerate(model_chain):
                if idx > 0:
                    rec.fallback_from = model_chain[idx - 1]
                rec.logical_model = model
                rec.provider = self._registry.get_config(model).provider
                norm = self._normalize(model, req, messages)

                # 用默认参数捕获当前迭代的 model/norm，避免闭包延迟绑定
                async def call(
                    attempt: int,
                    _model: str = model,
                    _norm: NormalizedRequest = norm,
                ) -> UnifiedResponse:
                    rec.attempt = attempt
                    t0 = time.perf_counter()
                    await self._limiter.acquire(_model, est_tokens)
                    rec.queue_ms = int((time.perf_counter() - t0) * 1000)
                    result = await self._invoke_adapter(_model, _norm, req.timeout_ms)
                    return self._to_response(result, rec, attempt)

                try:
                    resp = await with_retry(call, deadline=deadline, loop_time=loop.time)
                    succeeded_model = model
                    break
                except LLMError as e:
                    last_error = e
                    # 不允许 fallback 或已是最后一个候选 → 上抛
                    if not e.fallback_allowed or idx == len(model_chain) - 1:
                        raise

            # 理论上不会到这里（循环内要么 break 要么 raise）
            if succeeded_model is None or norm is None:
                raise last_error or LLMError(
                    LLMErrorCode.UNKNOWN, "无可用模型", stage=LLMStage.ROUTING
                )

            # 结构化输出校验 + 1 次修复（独立预算）；修复复用实际命中的模型
            if req.response_schema:
                serving_model = succeeded_model

                async def repair_invoke(
                    n: NormalizedRequest, r: UnifiedRequest, rc: CallRecord, attempt: int
                ) -> UnifiedResponse:
                    await self._limiter.acquire(
                        serving_model, self._estimate_tokens(n.messages, r.max_tokens)
                    )
                    result = await self._invoke_adapter(serving_model, n, r.timeout_ms)
                    return self._to_response(result, rc, attempt)

                resp = await self._validator.validate_and_repair(
                    resp, req, norm, rec, repair_invoke
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
            est_tokens = self._estimate_tokens(messages, req.max_tokens)

            model_chain = self._build_model_chain(req.model)
            timeout = httpx.Timeout(req.timeout_ms / 1000)

            last_error: Optional[LLMError] = None
            first_chunk = None
            adapter_iter = None

            for idx, model in enumerate(model_chain):
                if idx > 0:
                    rec.fallback_from = model_chain[idx - 1]
                rec.logical_model = model
                rec.provider = self._registry.get_config(model).provider
                norm = self._normalize(model, req, messages)

                async def open_stream(
                    attempt: int,
                    _model: str = model,
                    _norm: NormalizedRequest = norm,
                ):
                    rec.attempt = attempt
                    await self._limiter.acquire(_model, est_tokens)
                    adapter = self._registry.resolve(_model)
                    gen = adapter.chat_stream(_norm, timeout)
                    # 实际取首 chunk：确认连接成功 + 上游无错误（首 token 前可安全重试/fallback）
                    first = await gen.__anext__()
                    return first, gen

                try:
                    first_chunk, adapter_iter = await with_retry(
                        open_stream, deadline=deadline, loop_time=loop.time
                    )
                    break
                except LLMError as e:
                    last_error = e
                    if not e.fallback_allowed or idx == len(model_chain) - 1:
                        raise

            if first_chunk is None or adapter_iter is None:
                raise last_error or LLMError(
                    LLMErrorCode.UNKNOWN, "无可用模型", stage=LLMStage.ROUTING
                )

            # 首 chunk 成功 → 进入 STREAMING，后续不再重试、不切换模型
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
                rec.generation_ms = rec.total_latency_ms - rec.ttft_ms - (rec.queue_ms or 0)
                rec.is_final = True
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
    def _build_model_chain(self, primary: str) -> list[str]:
        """构建模型调用链：主模型 + 候选池，去重并过滤未注册模型。"""
        chain: list[str] = [primary]
        try:
            cfg = self._registry.get_config(primary)
        except LLMError:
            return chain
        for cand in cfg.candidates:
            if cand not in chain and cand in self._registry._configs:
                chain.append(cand)
        return chain

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

    def _normalize(self, model: str, req: UnifiedRequest, messages: list[UnifiedMessage]) -> NormalizedRequest:
        cfg = self._registry.get_config(model)
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

    async def _invoke_adapter(
        self, model: str, norm: NormalizedRequest, timeout_ms: int
    ) -> AdapterResult:
        adapter = self._registry.resolve(model)
        timeout = httpx.Timeout(timeout_ms / 1000)
        return await adapter.chat(norm, timeout)

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
        cfg = self._registry.get_config(rec.logical_model)
        price = cfg.price_per_million
        from decimal import Decimal

        rec.cost_input = Decimal(rec.input_tokens) / Decimal(1_000_000) * Decimal(price.input)
        rec.cost_output = Decimal(rec.output_tokens) / Decimal(1_000_000) * Decimal(price.output)
        rec.cost_cached = Decimal(rec.cached_tokens) / Decimal(1_000_000) * Decimal(price.cached)
        rec.cost_total = (rec.cost_input or 0) + (rec.cost_output or 0) + (rec.cost_cached or 0)
        await self._telemetry._persist(rec)
        self._telemetry._emit_metrics(rec, status="success")
