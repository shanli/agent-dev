"""适配器注册表：按逻辑模型 key 路由到对应适配器实例。"""

from __future__ import annotations

import os

from app.adapters.anthropic_messages import AnthropicMessagesAdapter
from app.adapters.base import LLMAdapter
from app.adapters.openai_responses import OpenAIResponsesAdapter
from app.core.config import ModelConfig
from app.core.errors import LLMError, LLMErrorCode, LLMStage


class AdapterRegistry:
    def __init__(self, configs: dict[str, ModelConfig]) -> None:
        self._adapters: dict[str, LLMAdapter] = {}
        self._configs = configs
        for key, cfg in configs.items():
            self._adapters[key] = self._build(cfg)

    def _build(self, cfg: ModelConfig) -> LLMAdapter:
        api_key = os.getenv(cfg.api_key_env, "")
        if cfg.provider == "openai_responses":
            return OpenAIResponsesAdapter(api_key=api_key, base_url=cfg.base_url)
        if cfg.provider == "anthropic_messages":
            return AnthropicMessagesAdapter(api_key=api_key, base_url=cfg.base_url)
        raise ValueError(f"未知 provider: {cfg.provider}")

    def resolve(self, model: str) -> LLMAdapter:
        adapter = self._adapters.get(model)
        if adapter is None:
            raise LLMError(
                LLMErrorCode.NO_COMPATIBLE_MODEL,
                f"未注册的逻辑模型: {model}",
                stage=LLMStage.ROUTING,
            )
        return adapter

    def get_config(self, model: str) -> ModelConfig:
        cfg = self._configs.get(model)
        if cfg is None:
            raise LLMError(
                LLMErrorCode.NO_COMPATIBLE_MODEL,
                f"未注册的逻辑模型: {model}",
                stage=LLMStage.ROUTING,
            )
        return cfg

    async def aclose(self) -> None:
        for adapter in self._adapters.values():
            await adapter.aclose()
