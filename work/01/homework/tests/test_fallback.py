"""Fallback 路由 + 超时映射测试。

覆盖：
- 本地限流/上游 429 时切换候选模型
- 非 fallback 错误（如 401）不切换
- 超时映射到 UPSTREAM_TIMEOUT
"""

from __future__ import annotations

import httpx
import pytest
import respx
from sqlalchemy import select

from app.adapters.anthropic_messages import AnthropicMessagesAdapter
from app.adapters.base import NormalizedRequest
from app.adapters.openai_responses import OpenAIResponsesAdapter
from app.core.errors import LLMErrorCode
from app.core.types import UnifiedMessage, UnifiedRequest
from app.db.models import LLMCallRecord


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
# Fallback 路由
# ════════════════════════════════════════════════════════════
class TestFallbackRouting:
    @respx.mock
    @pytest.mark.asyncio
    async def test_upstream_429_switches_to_candidate(self, gateway, session_factory, sample_messages):
        """上游 429（fallback_allowed）应切换到候选模型并返回候选的响应。"""
        # 主模型 pro 返回 429
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(429, text="rate limited", headers={"retry-after": "1"})
        )
        # 候选 flash 返回成功
        respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_flash",
                    "model": "deepseek-v4-flash",
                    "content": [{"type": "text", "text": "from-flash"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                    "stop_reason": "end_turn",
                },
            )
        )

        req = UnifiedRequest(model="deepseek-v4-pro", messages=sample_messages)
        resp = await gateway.chat(req)

        assert resp.content == "from-flash"
        assert resp.actual_model == "deepseek-v4-flash"

        # 验证落库记录标记了 fallback_from
        async with session_factory() as s:
            row = (await s.execute(select(LLMCallRecord))).scalar_one()
            assert row.fallback_from == "deepseek-v4-pro"
            assert row.logical_model == "deepseek-v4-flash"

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_failed_does_not_fallback(self, gateway, sample_messages):
        """401（非 fallback_allowed）应直接上抛，不切换候选模型。"""
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(401, text="unauthorized")
        )
        # 候选也准备一个成功响应，但不应被调用
        flash_route = respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            return_value=httpx.Response(200, json={"id": "x", "content": [], "usage": {}, "stop_reason": "end_turn"})
        )

        req = UnifiedRequest(model="deepseek-v4-pro", messages=sample_messages)
        with pytest.raises(Exception) as exc_info:
            await gateway.chat(req)

        assert exc_info.value.code == LLMErrorCode.AUTH_FAILED
        assert flash_route.call_count == 0  # 候选未被调用

    @respx.mock
    @pytest.mark.asyncio
    async def test_all_candidates_exhausted_raises_last_error(self, gateway, sample_messages):
        """主模型与所有候选都返回 fallback_allowed 错误时，应抛出最后一个错误。"""
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(429, text="rate limited")
        )
        respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            return_value=httpx.Response(429, text="rate limited")
        )

        req = UnifiedRequest(model="deepseek-v4-pro", messages=sample_messages)
        with pytest.raises(Exception) as exc_info:
            await gateway.chat(req)

        assert exc_info.value.code == LLMErrorCode.UPSTREAM_RATE_LIMITED

    @respx.mock
    @pytest.mark.asyncio
    async def test_streaming_fallback_on_waiting_429(self, gateway, sample_messages):
        """流式 WAITING 期 429 应切换候选模型。"""
        import json

        # 主模型返回 429
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(429, text="rate limited")
        )
        # 候选 flash 返回成功流
        events = [
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "flash-stream"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        ]
        sse = b"".join(f"data: {json.dumps(ev)}\n\n".encode() for ev in events)
        respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            return_value=httpx.Response(200, content=sse)
        )

        req = UnifiedRequest(model="deepseek-v4-pro", messages=sample_messages, stream=True)
        chunks = []
        async for chunk in gateway.chat_stream(req):
            chunks.append(chunk)

        deltas = "".join(c.delta or "" for c in chunks if c.type == "text_delta")
        assert deltas == "flash-stream"
        assert any(c.type == "done" for c in chunks)


# ════════════════════════════════════════════════════════════
# 超时映射
# ════════════════════════════════════════════════════════════
class TestTimeoutMapping:
    @respx.mock
    @pytest.mark.asyncio
    async def test_openai_timeout_maps_to_upstream_timeout(self, openai_adapter, norm_request):
        """OpenAI 适配器：httpx 超时应映射为 UPSTREAM_TIMEOUT。"""
        respx.post("https://api.deepseek.com/v1/responses").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )
        with pytest.raises(Exception) as exc_info:
            await openai_adapter.chat(norm_request, httpx.Timeout(10))
        assert exc_info.value.code == LLMErrorCode.UPSTREAM_TIMEOUT
        assert exc_info.value.retryable is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_anthropic_timeout_maps_to_upstream_timeout(self, anthropic_adapter, sample_messages):
        """Anthropic 适配器：httpx 超时应映射为 UPSTREAM_TIMEOUT。"""
        from app.adapters.base import NormalizedRequest

        req = NormalizedRequest(actual_model="deepseek-v4-flash", messages=sample_messages)
        respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            side_effect=httpx.ConnectTimeout("timed out")
        )
        with pytest.raises(Exception) as exc_info:
            await anthropic_adapter.chat(req, httpx.Timeout(10))
        assert exc_info.value.code == LLMErrorCode.UPSTREAM_TIMEOUT
        assert exc_info.value.retryable is True
