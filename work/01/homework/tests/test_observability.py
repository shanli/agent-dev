"""可观测性测试：验证 Trace、Metrics、Cost Ledger 是否正确记录了每次调用的关键字段。"""

from __future__ import annotations

from decimal import Decimal

import pytest
from prometheus_client import generate_latest

from app.core.errors import LLMError, LLMErrorCode, LLMStage
from app.core.types import UnifiedRequest, UnifiedResponse, Usage
from app.db.models import LLMCallRecord


# ════════════════════════════════════════════════════════════
# Trace 记录
# ════════════════════════════════════════════════════════════
class TestTraceRecording:
    def test_start_generates_trace_and_call_id(self, telemetry):
        req = UnifiedRequest(model="deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}])
        rec = telemetry.start(req)
        assert rec.trace_id  # 非空
        assert rec.call_id  # 非空
        assert rec.logical_model == "deepseek-v4-pro"

    def test_start_preserves_upstream_trace_id(self, telemetry):
        req = UnifiedRequest(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
            trace_id="upstream-trace-123",
            run_id="run-1",
            step_id="step-1",
        )
        rec = telemetry.start(req)
        assert rec.trace_id == "upstream-trace-123"
        assert rec.run_id == "run-1"
        assert rec.step_id == "step-1"

    @pytest.mark.asyncio
    async def test_finalize_persists_success_record(self, telemetry, session_factory):
        from app.core.types import UnifiedMessage

        req = UnifiedRequest(model="deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}])
        rec = telemetry.start(req)
        resp = UnifiedResponse(
            id="resp-1",
            logical_model="deepseek-v4-pro",
            actual_model="deepseek-v4-pro",
            content="hello",
            usage=Usage(input_tokens=10, output_tokens=20, cached_tokens=5, reasoning_tokens=3),
            finish_reason="stop",
            total_latency_ms=500,
            ttft_ms=100,
            provider_request_id="req-abc",
        )
        await telemetry.finalize(rec, resp)

        # 验证落库
        async with session_factory() as s:
            from sqlalchemy import select

            result = await s.execute(select(LLMCallRecord).where(LLMCallRecord.call_id == rec.call_id))
            row = result.scalar_one()

            assert row.trace_id == rec.trace_id
            assert row.final_state == "success"
            assert row.is_final is True
            assert row.actual_model == "deepseek-v4-pro"
            assert row.input_tokens == 10
            assert row.output_tokens == 20
            assert row.cached_tokens == 5
            assert row.reasoning_tokens == 3
            assert row.finish_reason == "stop"
            assert row.provider_request_id == "req-abc"
            assert row.ttft_ms == 100
            assert row.total_latency_ms == 500

    @pytest.mark.asyncio
    async def test_fail_persists_error_record(self, telemetry, session_factory):
        req = UnifiedRequest(model="deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}])
        rec = telemetry.start(req)
        err = LLMError(
            LLMErrorCode.UPSTREAM_RATE_LIMITED,
            "rate limited",
            stage=LLMStage.PRE_FIRST_TOKEN,
            http_status=429,
            provider_request_id="req-xyz",
        )
        await telemetry.fail(rec, err)

        async with session_factory() as s:
            from sqlalchemy import select

            result = await s.execute(select(LLMCallRecord).where(LLMCallRecord.call_id == rec.call_id))
            row = result.scalar_one()

            assert row.final_state == "error"
            assert row.error_code == "UPSTREAM_RATE_LIMITED"
            assert row.http_status == 429
            assert row.provider_request_id == "req-xyz"


# ════════════════════════════════════════════════════════════
# Cost Ledger
# ════════════════════════════════════════════════════════════
class TestCostLedger:
    @pytest.mark.asyncio
    async def test_cost_calculation(self, telemetry, session_factory):
        """成本应按 models.yaml 定价计算：input=0.27, output=1.10, cached=0.07 每百万。"""
        req = UnifiedRequest(model="deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}])
        rec = telemetry.start(req)
        resp = UnifiedResponse(
            id="resp-1",
            logical_model="deepseek-v4-pro",
            actual_model="deepseek-v4-pro",
            content="hello",
            usage=Usage(input_tokens=1_000_000, output_tokens=1_000_000, cached_tokens=1_000_000),
            finish_reason="stop",
            total_latency_ms=100,
        )
        await telemetry.finalize(rec, resp)

        async with session_factory() as s:
            from sqlalchemy import select

            result = await s.execute(select(LLMCallRecord).where(LLMCallRecord.call_id == rec.call_id))
            row = result.scalar_one()

            # 1M tokens * price_per_million
            assert row.cost_input == Decimal("0.27")
            assert row.cost_output == Decimal("1.10")
            assert row.cost_cached == Decimal("0.07")
            assert row.cost_total == Decimal("1.44")
            assert row.currency == "USD"


# ════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════
class TestMetrics:
    @pytest.mark.asyncio
    async def test_success_emits_metrics(self, telemetry):
        from prometheus_client.parser import text_string_to_metric_families

        req = UnifiedRequest(model="deepseek-v4-pro", messages=[{"role": "user", "content": "hi"}])
        rec = telemetry.start(req)
        resp = UnifiedResponse(
            id="resp-1",
            logical_model="deepseek-v4-pro",
            actual_model="deepseek-v4-pro",
            content="hello",
            usage=Usage(input_tokens=100, output_tokens=50),
            finish_reason="stop",
            total_latency_ms=200,
        )
        await telemetry.finalize(rec, resp)

        metrics_text = generate_latest().decode()
        families = {f.name: f for f in text_string_to_metric_families(metrics_text)}

        # 调用数指标存在且有 success 样本（counter family 名会去掉 _total 后缀）
        calls = families["llm_calls"]
        success_samples = [s for s in calls.samples if s.labels.get("status") == "success"]
        assert len(success_samples) > 0

        # token 指标
        tokens = families["llm_tokens"]
        input_samples = [s for s in tokens.samples if s.labels.get("direction") == "input"]
        assert len(input_samples) > 0

        # 延时指标
        assert "llm_latency_seconds" in families
