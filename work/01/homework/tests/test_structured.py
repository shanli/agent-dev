"""Structured Output 测试：验证 JSON 提取、Schema 校验、修复尝试次数、校验失败时的错误返回。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.errors import LLMErrorCode
from app.core.types import UnifiedRequest, UnifiedResponse, Usage
from app.gateway.structured.validator import MAX_REPAIRS, StructuredValidator


@pytest.fixture
def validator():
    return StructuredValidator()


@pytest.fixture
def sample_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }


@pytest.fixture
def sample_request(sample_schema):
    return UnifiedRequest(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "extract"}],
        response_schema=sample_schema,
    )


# ════════════════════════════════════════════════════════════
# Schema 校验
# ════════════════════════════════════════════════════════════
class TestSchemaValidation:
    def test_valid_json_passes(self, validator, sample_schema):
        validator.validate('{"name": "Alice", "age": 30}', sample_schema)

    def test_invalid_json_raises(self, validator, sample_schema):
        with pytest.raises(Exception) as exc_info:
            validator.validate("not a json", sample_schema)
        assert exc_info.value.code == LLMErrorCode.SCHEMA_VALIDATION_FAILED

    def test_type_mismatch_raises(self, validator, sample_schema):
        with pytest.raises(Exception) as exc_info:
            validator.validate('{"name": "Alice", "age": "thirty"}', sample_schema)
        assert exc_info.value.code == LLMErrorCode.SCHEMA_VALIDATION_FAILED

    def test_missing_required_field_raises(self, validator, sample_schema):
        with pytest.raises(Exception) as exc_info:
            validator.validate('{"name": "Alice"}', sample_schema)
        assert exc_info.value.code == LLMErrorCode.SCHEMA_VALIDATION_FAILED


# ════════════════════════════════════════════════════════════
# 修复流程
# ════════════════════════════════════════════════════════════
class TestValidateAndRepair:
    def _make_response(self, content: str) -> UnifiedResponse:
        return UnifiedResponse(
            id="resp-1",
            logical_model="deepseek-v4-pro",
            actual_model="deepseek-v4-pro",
            content=content,
            usage=Usage(input_tokens=5, output_tokens=10),
            finish_reason="stop",
            total_latency_ms=100,
        )

    @pytest.mark.asyncio
    async def test_valid_output_no_repair(self, validator, sample_request):
        """合法输出不触发修复。"""
        from app.adapters.base import NormalizedRequest
        from app.observability.telemetry import CallRecord

        rec = CallRecord(trace_id="t", call_id="c", logical_model="m")
        resp = self._make_response('{"name": "Alice", "age": 30}')
        invoke = AsyncMock()

        result = await validator.validate_and_repair(
            resp, sample_request, NormalizedRequest(actual_model="m", messages=[]), rec, invoke
        )

        assert result.content == '{"name": "Alice", "age": 30}'
        assert rec.validation_result == "passed"
        invoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_output_triggers_one_repair(self, validator, sample_request):
        """非法输出应触发 1 次修复，修复成功则返回。"""
        from app.adapters.base import NormalizedRequest
        from app.observability.telemetry import CallRecord

        rec = CallRecord(trace_id="t", call_id="c", logical_model="m")
        bad_resp = self._make_response("not json")
        good_resp = self._make_response('{"name": "Bob", "age": 25}')
        invoke = AsyncMock(return_value=good_resp)

        result = await validator.validate_and_repair(
            bad_resp, sample_request, NormalizedRequest(actual_model="m", messages=[]), rec, invoke
        )

        assert result.content == '{"name": "Bob", "age": 25}'
        assert rec.validation_result == "repair_retried"
        assert invoke.await_count == 1  # 恰好修复 1 次

    @pytest.mark.asyncio
    async def test_repair_still_failed_raises(self, validator, sample_request):
        """修复后仍失败应抛 SCHEMA_VALIDATION_FAILED。"""
        from app.adapters.base import NormalizedRequest
        from app.observability.telemetry import CallRecord

        rec = CallRecord(trace_id="t", call_id="c", logical_model="m")
        bad_resp = self._make_response("not json")
        still_bad = self._make_response("still not json")
        invoke = AsyncMock(return_value=still_bad)

        with pytest.raises(Exception) as exc_info:
            await validator.validate_and_repair(
                bad_resp, sample_request, NormalizedRequest(actual_model="m", messages=[]), rec, invoke
            )

        assert exc_info.value.code == LLMErrorCode.SCHEMA_VALIDATION_FAILED
        assert rec.validation_result == "failed"
        assert invoke.await_count == MAX_REPAIRS  # 修复预算用尽

    @pytest.mark.asyncio
    async def test_repair_budget_is_one(self, validator, sample_request):
        """修复预算 = 1（独立于网络重试预算）。"""
        assert MAX_REPAIRS == 1
