"""Anthropic Messages 协议适配器（用于 DeepSeek v4 flash）。

参考：https://docs.anthropic.com/en/api/messages
- POST {base_url}/v1/messages
- headers: x-api-key, anthropic-version: 2023-06-01
- system 字段代替 system message
- max_tokens 必填
- usage: input_tokens / output_tokens / cache_read_input_tokens → cached_tokens
- stop_reason: end_turn→stop, max_tokens→length, refusal→content_filter, tool_use→stop
- 结构化输出：用 tool_choice 强制 tool_use，从 content[].input 提取 JSON
- 流式：content_block_delta (text_delta), message_delta (stop_reason), message_stop
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.adapters.base import AdapterChunk, AdapterResult, LLMAdapter, NormalizedRequest
from app.adapters.sse import iter_sse_data
from app.core.errors import LLMError, LLMErrorCode, LLMStage
from app.core.types import Usage


class AnthropicMessagesAdapter(LLMAdapter):
    provider = "anthropic_messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_payload(self, req: NormalizedRequest, *, stream: bool) -> dict:
        system = "\n\n".join(m.content for m in req.messages if m.role == "system")
        payload: dict = {
            "model": req.actual_model,
            "max_tokens": req.max_tokens or 4096,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in req.messages
                if m.role != "system"
            ],
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.response_schema:
            # 用 tool_use 强制结构化输出
            payload["tools"] = [
                {
                    "name": req.schema_name,
                    "description": "按 schema 输出结构化结果",
                    "input_schema": req.response_schema,
                }
            ]
            payload["tool_choice"] = {"type": "tool", "name": req.schema_name}
        return payload

    async def chat(self, req: NormalizedRequest, timeout: httpx.Timeout) -> AdapterResult:
        payload = self._build_payload(req, stream=False)
        try:
            resp = await self._client.post(
                "/v1/messages", json=payload, headers=self._headers(), timeout=timeout
            )
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMError(
                LLMErrorCode.UPSTREAM_TIMEOUT,
                str(e),
                stage=LLMStage.PRE_FIRST_TOKEN,
                cause=e,
            )
        except httpx.HTTPStatusError as e:
            raise self._map_http_error(e)
        except httpx.TransportError as e:
            raise LLMError(
                LLMErrorCode.NETWORK_ERROR,
                str(e),
                stage=LLMStage.CONNECTION,
                cause=e,
            )

        data = resp.json()
        usage = self._normalize_usage(data.get("usage") or {})

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                # 结构化输出：tool 的 input 即目标 JSON
                content = json.dumps(block.get("input", {}), ensure_ascii=False)

        stop_reason = data.get("stop_reason")
        finish_reason = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "stop",
            "stop_sequence": "stop",
            "refusal": "content_filter",
        }.get(stop_reason, "stop")

        return AdapterResult(
            request_id=resp.headers.get("request-id") or data.get("id"),
            actual_model=data.get("model", req.actual_model),
            content=content,
            usage=usage,
            finish_reason=finish_reason,
            raw_http_status=resp.status_code,
        )

    async def chat_stream(
        self, req: NormalizedRequest, timeout: httpx.Timeout
    ) -> AsyncIterator[AdapterChunk]:
        payload = self._build_payload(req, stream=True)
        try:
            async with self._client.stream(
                "POST", "/v1/messages", json=payload, headers=self._headers(), timeout=timeout
            ) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    raise self._map_http_error(
                        httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )
                    )
                async for event in iter_sse_data(resp):
                    ev_type = event.get("type")
                    if ev_type == "content_block_delta":
                        delta = (event.get("delta") or {}).get("text", "")
                        if delta:
                            yield AdapterChunk(type="text_delta", delta=delta)
                    elif ev_type == "message_delta":
                        usage = self._normalize_usage(event.get("usage") or {})
                        if usage.input_tokens or usage.output_tokens:
                            yield AdapterChunk(type="usage", usage=usage)
                        stop_reason = (event.get("delta") or {}).get("stop_reason")
                        if stop_reason:
                            fr = {
                                "end_turn": "stop",
                                "max_tokens": "length",
                                "tool_use": "stop",
                                "refusal": "content_filter",
                            }.get(stop_reason, "stop")
                            yield AdapterChunk(type="done", finish_reason=fr)
        except httpx.TimeoutException as e:
            raise LLMError(
                LLMErrorCode.UPSTREAM_TIMEOUT,
                str(e),
                stage=LLMStage.PRE_FIRST_TOKEN,
                cause=e,
            )
        except httpx.TransportError as e:
            raise LLMError(
                LLMErrorCode.NETWORK_ERROR,
                str(e),
                stage=LLMStage.PRE_FIRST_TOKEN,
                cause=e,
            )

    @staticmethod
    def _normalize_usage(u: dict) -> Usage:
        return Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cached_tokens=u.get("cache_read_input_tokens", 0),
            reasoning_tokens=0,
        )

    def _map_http_error(self, e: httpx.HTTPStatusError) -> LLMError:
        status = e.response.status_code
        rid = e.response.headers.get("request-id")
        body_text = e.response.text
        if status in (401, 403):
            return LLMError(
                LLMErrorCode.AUTH_FAILED,
                body_text,
                stage=LLMStage.AUTH,
                http_status=status,
                provider_request_id=rid,
            )
        if status == 400:
            return LLMError(
                LLMErrorCode.INVALID_REQUEST,
                body_text,
                stage=LLMStage.VALIDATION,
                http_status=status,
                provider_request_id=rid,
            )
        if status == 429:
            retry_after = e.response.headers.get("retry-after")
            retry_after_ms = int(float(retry_after) * 1000) if retry_after else None
            return LLMError(
                LLMErrorCode.UPSTREAM_RATE_LIMITED,
                body_text,
                stage=LLMStage.PRE_FIRST_TOKEN,
                http_status=status,
                provider_request_id=rid,
                retry_after_ms=retry_after_ms,
            )
        if status == 529:
            return LLMError(
                LLMErrorCode.UPSTREAM_OVERLOADED,
                body_text,
                stage=LLMStage.PRE_FIRST_TOKEN,
                http_status=status,
                provider_request_id=rid,
            )
        if status >= 500:
            return LLMError(
                LLMErrorCode.UPSTREAM_UNAVAILABLE,
                body_text,
                stage=LLMStage.PRE_FIRST_TOKEN,
                http_status=status,
                provider_request_id=rid,
            )
        return LLMError(
            LLMErrorCode.UNKNOWN,
            body_text,
            stage=LLMStage.PRE_FIRST_TOKEN,
            http_status=status,
            provider_request_id=rid,
        )
