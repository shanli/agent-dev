"""OpenAI Responses 协议适配器（用于 DeepSeek v4 pro）。

参考：https://platform.openai.com/docs/api-reference/responses
- POST {base_url}/responses
- instructions 代替 system message
- 响应 output[].content[].text 聚合为 content
- usage: input_tokens_details.cached_tokens / output_tokens_details.reasoning_tokens
- 流式：response.output_text.delta 事件
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from app.adapters.base import AdapterChunk, AdapterResult, LLMAdapter, NormalizedRequest
from app.adapters.sse import iter_sse_data
from app.core.errors import LLMError, LLMErrorCode, LLMStage
from app.core.types import Usage


class OpenAIResponsesAdapter(LLMAdapter):
    provider = "openai_responses"

    # ── 请求构建 ──
    def _build_payload(self, req: NormalizedRequest, *, stream: bool) -> dict:
        system = [m.content for m in req.messages if m.role == "system"]
        payload: dict = {
            "model": req.actual_model,
            "input": [
                {"role": m.role, "content": m.content}
                for m in req.messages
                if m.role != "system"
            ],
            "stream": stream,
        }
        if system:
            payload["instructions"] = "\n\n".join(system)
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens:
            payload["max_output_tokens"] = req.max_tokens
        if req.response_schema:
            # OpenAI Responses 原生 JSON Schema 约束
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": req.schema_name,
                    "strict": True,
                    "schema": req.response_schema,
                }
            }
        return payload

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    # ── 非流式 ──
    async def chat(self, req: NormalizedRequest, timeout: httpx.Timeout) -> AdapterResult:
        payload = self._build_payload(req, stream=False)
        try:
            resp = await self._client.post(
                "/responses", json=payload, headers=self._headers(), timeout=timeout
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
        for output in data.get("output", []):
            if output.get("type") != "message":
                continue
            for c in output.get("content", []):
                if c.get("type") == "output_text":
                    content += c.get("text", "")

        incomplete = data.get("incomplete_details") or {}
        if incomplete.get("reason") == "max_output_tokens":
            finish_reason = "length"
        elif data.get("incomplete_details", {}).get("reason") == "content_filter":
            finish_reason = "content_filter"
        else:
            finish_reason = "stop"

        return AdapterResult(
            request_id=resp.headers.get("x-request-id") or data.get("id"),
            actual_model=data.get("model", req.actual_model),
            content=content,
            usage=usage,
            finish_reason=finish_reason,
            raw_http_status=resp.status_code,
        )

    # ── 流式 ──
    async def chat_stream(
        self, req: NormalizedRequest, timeout: httpx.Timeout
    ) -> AsyncIterator[AdapterChunk]:
        payload = self._build_payload(req, stream=True)
        try:
            async with self._client.stream(
                "POST", "/responses", json=payload, headers=self._headers(), timeout=timeout
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
                    if ev_type == "response.output_text.delta":
                        yield AdapterChunk(type="text_delta", delta=event.get("delta", ""))
                    elif ev_type == "response.completed":
                        usage = self._normalize_usage(event.get("response", {}).get("usage") or {})
                        yield AdapterChunk(type="usage", usage=usage)
                        reason = event.get("response", {}).get("incomplete_details", {}).get("reason")
                        fr = "length" if reason == "max_output_tokens" else "stop"
                        yield AdapterChunk(type="done", finish_reason=fr)
                    elif ev_type == "response.failed":
                        err = event.get("response", {}).get("error", {})
                        raise LLMError(
                            LLMErrorCode.UNKNOWN,
                            str(err),
                            stage=LLMStage.PRE_FIRST_TOKEN,
                        )
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

    # ── 工具方法 ──
    @staticmethod
    def _normalize_usage(u: dict) -> Usage:
        return Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cached_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens", 0),
            reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
        )

    def _map_http_error(self, e: httpx.HTTPStatusError) -> LLMError:
        status = e.response.status_code
        rid = e.response.headers.get("x-request-id")
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
