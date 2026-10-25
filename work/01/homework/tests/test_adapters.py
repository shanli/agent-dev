"""Adapter Contract Test：验证每个 Adapter 的请求翻译和响应翻译是否符合协议。

用 respx MockTransport 驱动，不依赖真实供应商。
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.adapters.anthropic_messages import AnthropicMessagesAdapter
from app.adapters.base import NormalizedRequest
from app.adapters.openai_responses import OpenAIResponsesAdapter
from app.core.errors import LLMErrorCode, LLMStage
from app.core.types import UnifiedMessage, Usage


@pytest.fixture
def openai_adapter():
    return OpenAIResponsesAdapter(api_key="sk-test", base_url="https://api.deepseek.com/v1")


@pytest.fixture
def anthropic_adapter():
    return AnthropicMessagesAdapter(api_key="sk-test", base_url="https://api.deepseek.com/anthropic")


@pytest.fixture
def norm_request(sample_messages):
    return NormalizedRequest(
        actual_model="deepseek-v4-pro",
        messages=sample_messages,
        temperature=0.7,
        max_tokens=100,
    )


# ════════════════════════════════════════════════════════════
# OpenAI Responses Adapter
# ════════════════════════════════════════════════════════════
class TestOpenAIResponsesAdapter:
    def test_build_payload_system_to_instructions(self, openai_adapter, norm_request):
        """system 消息应翻译为 instructions 字段。"""
        payload = openai_adapter._build_payload(norm_request, stream=False)
        assert payload["instructions"] == "你是助手"
        assert all(m["role"] != "system" for m in payload["input"])

    def test_build_payload_stream_flag(self, openai_adapter, norm_request):
        payload = openai_adapter._build_payload(norm_request, stream=True)
        assert payload["stream"] is True

    def test_build_payload_temperature_and_max_tokens(self, openai_adapter, norm_request):
        payload = openai_adapter._build_payload(norm_request, stream=False)
        assert payload["temperature"] == 0.7
        assert payload["max_output_tokens"] == 100

    def test_build_payload_json_schema(self, openai_adapter, sample_messages):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        req = NormalizedRequest(
            actual_model="deepseek-v4-pro",
            messages=sample_messages,
            response_schema=schema,
        )
        payload = openai_adapter._build_payload(req, stream=False)
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["schema"] == schema
        assert payload["text"]["format"]["strict"] is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_success_parses_content_and_usage(self, openai_adapter, norm_request):
        """成功响应应正确聚合 content 并归一化 usage。"""
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp_123",
                    "model": "deepseek-v4-pro",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "你好！"}]}
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 8,
                        "input_tokens_details": {"cached_tokens": 2},
                        "output_tokens_details": {"reasoning_tokens": 3},
                    },
                },
                headers={"x-request-id": "req-abc"},
            )
        )
        result = await openai_adapter.chat(norm_request, httpx.Timeout(10))
        assert result.content == "你好！"
        assert result.actual_model == "deepseek-v4-pro"
        assert result.request_id == "req-abc"
        assert result.usage == Usage(input_tokens=10, output_tokens=8, cached_tokens=2, reasoning_tokens=3)
        assert result.finish_reason == "stop"

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_max_tokens_finish_reason_length(self, openai_adapter, norm_request):
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp_123",
                    "model": "deepseek-v4-pro",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            )
        )
        result = await openai_adapter.chat(norm_request, httpx.Timeout(10))
        assert result.finish_reason == "length"

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_401_maps_to_auth_failed(self, openai_adapter, norm_request):
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(401, text="unauthorized")
        )
        with pytest.raises(Exception) as exc_info:
            await openai_adapter.chat(norm_request, httpx.Timeout(10))
        err = exc_info.value
        assert err.code == LLMErrorCode.AUTH_FAILED
        assert err.stage == LLMStage.AUTH
        assert err.http_status == 401
        assert err.retryable is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_429_maps_to_upstream_rate_limited_with_retry_after(self, openai_adapter, norm_request):
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(429, text="rate limited", headers={"retry-after": "2"})
        )
        with pytest.raises(Exception) as exc_info:
            await openai_adapter.chat(norm_request, httpx.Timeout(10))
        err = exc_info.value
        assert err.code == LLMErrorCode.UPSTREAM_RATE_LIMITED
        assert err.stage == LLMStage.PRE_FIRST_TOKEN
        assert err.retry_after_ms == 2000
        assert err.retryable is True
        assert err.fallback_allowed is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_500_maps_to_upstream_unavailable(self, openai_adapter, norm_request):
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(503, text="overloaded")
        )
        with pytest.raises(Exception) as exc_info:
            await openai_adapter.chat(norm_request, httpx.Timeout(10))
        err = exc_info.value
        assert err.code == LLMErrorCode.UPSTREAM_UNAVAILABLE
        assert err.retryable is True


# ════════════════════════════════════════════════════════════
# Anthropic Messages Adapter
# ════════════════════════════════════════════════════════════
class TestAnthropicMessagesAdapter:
    def test_build_payload_system_field(self, anthropic_adapter, sample_messages):
        """system 消息应翻译为 system 字段（非 messages 内）。"""
        req = NormalizedRequest(
            actual_model="deepseek-v4-flash",
            messages=sample_messages,
            max_tokens=200,
        )
        payload = anthropic_adapter._build_payload(req, stream=False)
        assert payload["system"] == "你是助手"
        assert all(m["role"] != "system" for m in payload["messages"])
        assert payload["max_tokens"] == 200

    def test_build_payload_default_max_tokens(self, anthropic_adapter, sample_messages):
        req = NormalizedRequest(actual_model="deepseek-v4-flash", messages=sample_messages)
        payload = anthropic_adapter._build_payload(req, stream=False)
        assert payload["max_tokens"] == 4096  # 默认值

    def test_build_payload_tool_use_for_structured(self, anthropic_adapter, sample_messages):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        req = NormalizedRequest(
            actual_model="deepseek-v4-flash",
            messages=sample_messages,
            response_schema=schema,
        )
        payload = anthropic_adapter._build_payload(req, stream=False)
        assert payload["tools"][0]["input_schema"] == schema
        assert payload["tool_choice"] == {"type": "tool", "name": "structured_output"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_success_text_content(self, anthropic_adapter, sample_messages):
        req = NormalizedRequest(actual_model="deepseek-v4-flash", messages=sample_messages)
        respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_123",
                    "model": "deepseek-v4-flash",
                    "content": [{"type": "text", "text": "你好！"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 1},
                    "stop_reason": "end_turn",
                },
                headers={"request-id": "req-xyz"},
            )
        )
        result = await anthropic_adapter.chat(req, httpx.Timeout(10))
        assert result.content == "你好！"
        assert result.usage == Usage(input_tokens=10, output_tokens=5, cached_tokens=1, reasoning_tokens=0)
        assert result.finish_reason == "stop"
        assert result.request_id == "req-xyz"

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_tool_use_extracts_json(self, anthropic_adapter, sample_messages):
        """结构化输出：从 tool_use 的 input 提取 JSON。"""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        req = NormalizedRequest(
            actual_model="deepseek-v4-flash",
            messages=sample_messages,
            response_schema=schema,
        )
        respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_123",
                    "model": "deepseek-v4-flash",
                    "content": [{"type": "tool_use", "input": {"name": "Alice"}, "id": "tu_1"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "stop_reason": "tool_use",
                },
            )
        )
        result = await anthropic_adapter.chat(req, httpx.Timeout(10))
        import json

        assert json.loads(result.content) == {"name": "Alice"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_stop_reason_refusal_maps_to_content_filter(self, anthropic_adapter, sample_messages):
        req = NormalizedRequest(actual_model="deepseek-v4-flash", messages=sample_messages)
        respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_123",
                    "model": "deepseek-v4-flash",
                    "content": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "refusal",
                },
            )
        )
        result = await anthropic_adapter.chat(req, httpx.Timeout(10))
        assert result.finish_reason == "content_filter"

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_529_maps_to_overloaded(self, anthropic_adapter, sample_messages):
        req = NormalizedRequest(actual_model="deepseek-v4-flash", messages=sample_messages)
        respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            return_value=httpx.Response(529, text="overloaded")
        )
        with pytest.raises(Exception) as exc_info:
            await anthropic_adapter.chat(req, httpx.Timeout(10))
        err = exc_info.value
        assert err.code == LLMErrorCode.UPSTREAM_OVERLOADED
        assert err.retryable is True
