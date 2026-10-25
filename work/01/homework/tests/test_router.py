"""Router 测试：验证路由策略、能力匹配、错误码到 HTTP 状态映射。"""

from __future__ import annotations

import httpx
import pytest
import respx
from starlette.testclient import TestClient

from app.adapters.registry import AdapterRegistry
from app.core.errors import LLMErrorCode
from app.core.types import UnifiedMessage, UnifiedRequest


# ════════════════════════════════════════════════════════════
# AdapterRegistry 路由
# ════════════════════════════════════════════════════════════
class TestAdapterRegistry:
    def test_resolve_registered_model_returns_correct_adapter(self, registry):
        adapter = registry.resolve("deepseek-v4-pro")
        assert adapter.provider == "openai_responses"

        adapter = registry.resolve("deepseek-v4-flash")
        assert adapter.provider == "anthropic_messages"

    def test_resolve_unregistered_model_raises_no_compatible_model(self, registry):
        from app.core.errors import LLMError

        with pytest.raises(LLMError) as exc_info:
            registry.resolve("nonexistent-model")
        assert exc_info.value.code == LLMErrorCode.NO_COMPATIBLE_MODEL
        assert exc_info.value.retryable is False

    def test_get_config_returns_full_config(self, registry):
        cfg = registry.get_config("deepseek-v4-pro")
        assert cfg.actual_model == "deepseek-v4-pro"
        assert cfg.provider == "openai_responses"
        assert cfg.price_per_million.input == 0.27


# ════════════════════════════════════════════════════════════
# HTTP 路由 + 错误码映射
# ════════════════════════════════════════════════════════════
class TestHttpRouting:
    def test_unknown_model_returns_404(self, app):
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat",
                json={"model": "unknown-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "NO_COMPATIBLE_MODEL"

    def test_missing_messages_returns_400(self, app):
        with TestClient(app) as client:
            r = client.post("/v1/chat", json={"model": "deepseek-v4-pro"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_REQUEST"

    @respx.mock
    def test_successful_chat_returns_200(self, app):
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "model": "deepseek-v4-pro",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            )
        )
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat",
                json={"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["content"] == "hi"
        assert data["actual_model"] == "deepseek-v4-pro"
        assert data["usage"]["input_tokens"] == 5
        assert data["usage"]["output_tokens"] == 3

    @respx.mock
    def test_upstream_401_returns_401(self, app):
        respx.post("https://api.deepseek.com/v1/responses").mock(
            return_value=httpx.Response(401, text="unauthorized")
        )
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat",
                json={"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "AUTH_FAILED"
