"""配置加载：pydantic-settings + models.yaml。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PricePerMillion(BaseModel):
    input: float
    output: float
    cached: float = 0.0


class ModelConfig(BaseModel):
    """单个模型的完整配置。"""

    key: str
    provider: Literal["openai_responses", "anthropic_messages"]
    actual_model: str
    base_url: str
    api_key_env: str
    rpm: int
    tpm: int
    price_per_million: PricePerMillion
    candidates: list[str] = []  # 候选模型 key 列表，fallback 时按顺序切换


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://llm:llm@localhost:5432/llm"
    models_path: Path = Path("models.yaml")
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    def load_models(self) -> dict[str, ModelConfig]:
        raw = yaml.safe_load(self.models_path.read_text(encoding="utf-8"))["models"]
        return {k: ModelConfig(key=k, **v) for k, v in raw.items()}


settings = Settings()
