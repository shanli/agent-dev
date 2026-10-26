"""限流测试：验证双令牌桶（RPM + TPM）的耗尽、retry_after、未注册模型与补充行为。"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.config import ModelConfig, PricePerMillion
from app.core.errors import LLMErrorCode
from app.gateway.resilience.ratelimit import ModelRateLimiter, TokenBucket


def _cfg(rpm: int, tpm: int) -> ModelConfig:
    return ModelConfig(
        key="m",
        provider="openai_responses",
        actual_model="m",
        base_url="https://x",
        api_key_env="K",
        rpm=rpm,
        tpm=tpm,
        price_per_million=PricePerMillion(input=0.1, output=0.1, cached=0.0),
    )


# ════════════════════════════════════════════════════════════
# TokenBucket 单元行为
# ════════════════════════════════════════════════════════════
class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_acquire_within_capacity_succeeds(self):
        b = TokenBucket(capacity=5, refill_per_sec=1.0)
        assert await b.try_acquire(3) is None
        assert await b.try_acquire(2) is None

    @pytest.mark.asyncio
    async def test_acquire_exceeding_capacity_returns_wait(self):
        b = TokenBucket(capacity=2, refill_per_sec=1.0)
        wait = await b.try_acquire(5)
        assert wait is not None
        assert wait > 0

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        """等待后令牌应得到补充。"""
        b = TokenBucket(capacity=10, refill_per_sec=100.0)
        # 先耗尽
        await b.try_acquire(10)
        assert await b.try_acquire(1) is not None
        # 等 50ms，按 100/s 应补充约 5 个
        await asyncio.sleep(0.05)
        assert await b.try_acquire(1) is None


# ════════════════════════════════════════════════════════════
# ModelRateLimiter 集成行为
# ════════════════════════════════════════════════════════════
class TestModelRateLimiter:
    def _limiter(self, rpm: int = 2, tpm: int = 100) -> ModelRateLimiter:
        return ModelRateLimiter({"m": _cfg(rpm=rpm, tpm=tpm)})

    @pytest.mark.asyncio
    async def test_rpm_exhausted_raises_rate_limited(self):
        """RPM 桶耗尽后，新请求应抛 RATE_LIMITED(429)。"""
        limiter = self._limiter(rpm=2, tpm=1000)
        await limiter.acquire("m", 1)
        await limiter.acquire("m", 1)
        with pytest.raises(Exception) as exc_info:
            await limiter.acquire("m", 1)
        assert exc_info.value.code == LLMErrorCode.RATE_LIMITED
        assert exc_info.value.retryable is False
        assert exc_info.value.fallback_allowed is True  # 本地限流允许转投候选池

    @pytest.mark.asyncio
    async def test_tpm_exhausted_raises_rate_limited(self):
        """TPM 桶耗尽（估算 token 超过剩余配额）应抛 RATE_LIMITED。"""
        limiter = self._limiter(rpm=1000, tpm=10)
        # 一次请求扣 20 token，超过 TPM 容量 10
        with pytest.raises(Exception) as exc_info:
            await limiter.acquire("m", 20)
        assert exc_info.value.code == LLMErrorCode.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_rate_limited_carries_retry_after_ms(self):
        """限流错误应携带 retry_after_ms，便于调用方退避。"""
        limiter = self._limiter(rpm=1, tpm=1000)
        await limiter.acquire("m", 1)
        with pytest.raises(Exception) as exc_info:
            await limiter.acquire("m", 1)
        assert exc_info.value.retry_after_ms is not None
        assert exc_info.value.retry_after_ms > 0

    @pytest.mark.asyncio
    async def test_unregistered_model_raises_no_compatible_model(self):
        """未注册的模型应抛 NO_COMPATIBLE_MODEL，不走限流逻辑。"""
        limiter = self._limiter()
        with pytest.raises(Exception) as exc_info:
            await limiter.acquire("nonexistent", 1)
        assert exc_info.value.code == LLMErrorCode.NO_COMPATIBLE_MODEL
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_refill_allows_retry_after_wait(self):
        """等待足够时间后，限流应解除（令牌补充）。"""
        limiter = self._limiter(rpm=10, tpm=1000)  # refill = 10/60 ≈ 0.167/s
        await limiter.acquire("m", 1)
        # 手动把桶掏空并置零更新时间，强制走补充路径
        bucket = limiter._buckets["m"][0]
        bucket.tokens = 0
        bucket.refill = 1000.0  # 极快补充
        bucket.updated = time.monotonic()
        await asyncio.sleep(0.01)
        # 现在应能成功
        await limiter.acquire("m", 1)
