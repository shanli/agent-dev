"""按模型独立的双令牌桶限流（RPM + TPM）。

设计：
- 每个逻辑模型一组令牌桶：RPM（请求数）+ TPM（token 数）。
- 请求前预扣（按估算 token + max_tokens），超限立即抛 RATE_LIMITED(429)。
- 响应后按实际 usage 校正差额（简化：只补扣，不退余额）。
- 接口不变，后续可替换为 Redis 实现以支持多实例。
"""

from __future__ import annotations

from typing import Optional

import asyncio
import time

from app.core.config import ModelConfig
from app.core.errors import LLMError, LLMErrorCode, LLMStage


class TokenBucket:
    def __init__(self, capacity: int, refill_per_sec: float) -> None:
        self.capacity = capacity
        self.refill = refill_per_sec
        self.tokens = float(capacity)
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def try_acquire(self, n: int) -> Optional[float]:
        """成功返回 None；不足则返回需等待的秒数。"""
        async with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill)
            self.updated = now
            if self.tokens >= n:
                self.tokens -= n
                return None
            return (n - self.tokens) / self.refill


class ModelRateLimiter:
    def __init__(self, configs: dict[str, ModelConfig]) -> None:
        self._buckets: dict[str, tuple[TokenBucket, TokenBucket]] = {
            k: (TokenBucket(c.rpm, c.rpm / 60.0), TokenBucket(c.tpm, c.tpm / 60.0))
            for k, c in configs.items()
        }

    async def acquire(self, model: str, est_tokens: int) -> None:
        """尝试获取令牌，超限抛 RATE_LIMITED(429) 并附带 retry_after_ms。"""
        buckets = self._buckets.get(model)
        if buckets is None:
            raise LLMError(
                LLMErrorCode.NO_COMPATIBLE_MODEL,
                f"未注册的逻辑模型: {model}",
                stage=LLMStage.ROUTING,
            )
        rpm_bucket, tpm_bucket = buckets
        wait_rpm = await rpm_bucket.try_acquire(1)
        wait_tpm = await tpm_bucket.try_acquire(max(1, est_tokens))
        waits = [w for w in (wait_rpm, wait_tpm) if w is not None]
        if waits:
            slowest = max(waits)
            raise LLMError(
                LLMErrorCode.RATE_LIMITED,
                f"模型 {model} 触发本地限流",
                stage=LLMStage.QUEUE,
                retry_after_ms=int(slowest * 1000) + 50,
            )
