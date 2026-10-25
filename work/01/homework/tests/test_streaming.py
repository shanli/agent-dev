"""Streaming 测试：验证 SSE 事件序列的正确性。

覆盖：delta 顺序、usage 事件、终态、错误中断行为。
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.adapters.base import NormalizedRequest
from app.core.types import UnifiedMessage


def _sse_lines(events: list[dict]) -> bytes:
    """把事件列表编码为 SSE 字节流。"""
    import json

    return b"".join(f"data: {json.dumps(ev)}\n\n".encode() for ev in events)


# ════════════════════════════════════════════════════════════
# OpenAI Responses 流式
# ════════════════════════════════════════════════════════════
class TestOpenAIStreaming:
    @pytest.fixture
    def adapter(self):
        from app.adapters.openai_responses import OpenAIResponsesAdapter

        return OpenAIResponsesAdapter(api_key="sk-test", base_url="https://api.deepseek.com/v1")

    @pytest.fixture
    def norm_req(self, sample_messages):
        return NormalizedRequest(actual_model="deepseek-v4-pro", messages=sample_messages)

    @respx.mock
    @pytest.mark.asyncio
    async def test_delta_order_and_done(self, adapter, norm_req):
        """delta 应按顺序产出，最后是 done。"""
        events = [
            {"type": "response.output_text.delta", "delta": "你"},
            {"type": "response.output_text.delta", "delta": "好"},
            {"type": "response.output_text.delta", "delta": "！"},
            {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                    "incomplete_details": {},
                },
            },
        ]
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(200, content=_sse_lines(events))
        )

        chunks = []
        async for chunk in adapter.chat_stream(norm_req, httpx.Timeout(10)):
            chunks.append(chunk)

        # 前三个是 text_delta，顺序正确
        assert [c.type for c in chunks[:3]] == ["text_delta"] * 3
        assert "".join(c.delta for c in chunks[:3]) == "你好！"
        # 倒数第二个是 usage
        assert chunks[-2].type == "usage"
        assert chunks[-2].usage.input_tokens == 5
        # 最后是 done
        assert chunks[-1].type == "done"
        assert chunks[-1].finish_reason == "stop"

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_429_raises_before_first_chunk(self, adapter, norm_req):
        """首 token 前的 429 应抛异常（可重试），不产出 chunk。"""
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(429, text="rate limited", headers={"retry-after": "1"})
        )

        chunks = []
        with pytest.raises(Exception) as exc_info:
            async for chunk in adapter.chat_stream(norm_req, httpx.Timeout(10)):
                chunks.append(chunk)

        assert chunks == []  # 没有任何 chunk 产出
        assert exc_info.value.code.value == "UPSTREAM_RATE_LIMITED"


# ════════════════════════════════════════════════════════════
# Anthropic Messages 流式
# ════════════════════════════════════════════════════════════
class TestAnthropicStreaming:
    @pytest.fixture
    def adapter(self):
        from app.adapters.anthropic_messages import AnthropicMessagesAdapter

        return AnthropicMessagesAdapter(api_key="sk-test", base_url="https://api.deepseek.com/anthropic")

    @pytest.fixture
    def norm_req(self, sample_messages):
        return NormalizedRequest(actual_model="deepseek-v4-flash", messages=sample_messages)

    @respx.mock
    @pytest.mark.asyncio
    async def test_delta_usage_done_sequence(self, adapter, norm_req):
        events = [
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "!"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        ]
        respx.post("https://api.deepseek.com/anthropic/v1/messages").mock(
            return_value=httpx.Response(200, content=_sse_lines(events))
        )

        chunks = []
        async for chunk in adapter.chat_stream(norm_req, httpx.Timeout(10)):
            chunks.append(chunk)

        assert [c.type for c in chunks] == ["text_delta", "text_delta", "usage", "done"]
        assert chunks[0].delta == "Hi"
        assert chunks[1].delta == "!"
        assert chunks[2].usage.output_tokens == 2
        assert chunks[3].finish_reason == "stop"


# ════════════════════════════════════════════════════════════
# Gateway 流式：错误中断行为
# ════════════════════════════════════════════════════════════
class TestGatewayStreamingErrorHandling:
    @respx.mock
    @pytest.mark.asyncio
    async def test_waiting_stage_error_propagates(self, gateway, sample_messages):
        """首 token 前的错误（如 401）应直接上抛，不发 error chunk。"""
        from app.core.errors import LLMErrorCode
        from app.core.types import UnifiedRequest

        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(401, text="unauthorized")
        )

        req = UnifiedRequest(model="deepseek-v4-pro", messages=sample_messages, stream=True)
        chunks = []
        with pytest.raises(Exception) as exc_info:
            async for chunk in gateway.chat_stream(req):
                chunks.append(chunk)

        assert chunks == []
        assert exc_info.value.code == LLMErrorCode.AUTH_FAILED

    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_stream_emits_deltas(self, gateway, sample_messages):
        """成功流式应按顺序产出 delta。"""
        import json

        from app.core.types import UnifiedRequest

        events = [
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": " World"},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 3, "output_tokens": 2}},
            },
        ]
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(200, content=_sse_lines(events))
        )

        req = UnifiedRequest(model="deepseek-v4-pro", messages=sample_messages, stream=True)
        chunks = []
        async for chunk in gateway.chat_stream(req):
            chunks.append(chunk)

        types = [c.type for c in chunks]
        assert "text_delta" in types
        assert "done" in types
        # 所有 delta 拼接
        deltas = "".join(c.delta or "" for c in chunks if c.type == "text_delta")
        assert deltas == "Hello World"
