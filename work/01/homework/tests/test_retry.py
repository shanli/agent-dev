"""重试测试：用 Mock 模拟超时和错误，验证重试次数和退避策略。

关键约束：不能真的 sleep，必须 mock asyncio.sleep。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.core.errors import LLMError, LLMErrorCode, LLMStage
from app.gateway.resilience.retry import MAX_RETRIES, with_retry


@pytest.fixture(autouse=True)
def mock_sleep():
    """全局 mock asyncio.sleep，避免测试真的等待。"""
    with patch("app.gateway.resilience.retry.asyncio.sleep", new_callable=AsyncMock) as m:
        yield m


class TestRetryExecutor:
    @pytest.mark.asyncio
    async def test_retryable_error_retries_max_times(self, mock_sleep):
        """可重试错误应重试 MAX_RETRIES 次（共 MAX_RETRIES+1 次尝试）。"""
        calls = {"count": 0}

        async def fn(attempt: int):
            calls["count"] += 1
            raise LLMError(
                LLMErrorCode.NETWORK_ERROR,
                "network error",
                stage=LLMStage.PRE_FIRST_TOKEN,
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 60  # 充足预算

        with pytest.raises(LLMError) as exc_info:
            await with_retry(fn, deadline=deadline, loop_time=loop.time)

        # 首次 + MAX_RETRIES 次重试
        assert calls["count"] == MAX_RETRIES + 1
        assert exc_info.value.code == LLMErrorCode.NETWORK_ERROR
        # 退避应被调用 MAX_RETRIES 次
        assert mock_sleep.await_count == MAX_RETRIES

    @pytest.mark.asyncio
    async def test_non_retryable_error_no_retry(self, mock_sleep):
        """不可重试错误应立即上抛，不重试。"""
        calls = {"count": 0}

        async def fn(attempt: int):
            calls["count"] += 1
            raise LLMError(
                LLMErrorCode.AUTH_FAILED,
                "auth failed",
                stage=LLMStage.AUTH,
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 60

        with pytest.raises(LLMError) as exc_info:
            await with_retry(fn, deadline=deadline, loop_time=loop.time)

        assert calls["count"] == 1
        assert mock_sleep.await_count == 0
        assert exc_info.value.code == LLMErrorCode.AUTH_FAILED

    @pytest.mark.asyncio
    async def test_success_on_second_attempt(self, mock_sleep):
        """首次失败、第二次成功应返回成功结果。"""
        calls = {"count": 0}

        async def fn(attempt: int):
            calls["count"] += 1
            if attempt == 1:
                raise LLMError(
                    LLMErrorCode.NETWORK_ERROR,
                    "transient",
                    stage=LLMStage.PRE_FIRST_TOKEN,
                )
            return "success"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 60

        result = await with_retry(fn, deadline=deadline, loop_time=loop.time)
        assert result == "success"
        assert calls["count"] == 2
        assert mock_sleep.await_count == 1  # 只退避一次

    @pytest.mark.asyncio
    async def test_deadline_exceeded_raises_before_attempt(self):
        """截止时间预算不足时，直接抛 DEADLINE_EXCEEDED，不发起调用。"""
        calls = {"count": 0}

        async def fn(attempt: int):
            calls["count"] += 1
            return "ok"

        loop = asyncio.get_running_loop()
        deadline = loop.time() - 1  # 已过期

        with pytest.raises(LLMError) as exc_info:
            await with_retry(fn, deadline=deadline, loop_time=loop.time)

        assert calls["count"] == 0
        assert exc_info.value.code == LLMErrorCode.DEADLINE_EXCEEDED

    @pytest.mark.asyncio
    async def test_respects_upstream_retry_after(self, mock_sleep):
        """上游返回 retry-after 时，退避时间应取 max(retry_after, 指数退避)。"""

        async def fn(attempt: int):
            raise LLMError(
                LLMErrorCode.UPSTREAM_RATE_LIMITED,
                "rate limited",
                stage=LLMStage.PRE_FIRST_TOKEN,
                retry_after_ms=5000,  # 上游要求 5 秒后重试
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 60

        with pytest.raises(LLMError):
            await with_retry(fn, deadline=deadline, loop_time=loop.time)

        # 退避时间应 >= 5000ms（上游 retry-after）
        for call in mock_sleep.await_args_list:
            sleep_seconds = call.args[0]
            assert sleep_seconds >= 5.0

    @pytest.mark.asyncio
    async def test_content_filtered_not_retried_not_fallback(self, mock_sleep):
        """内容拒答：不重试，且 fallback_allowed=False（红线规则）。"""
        calls = {"count": 0}

        async def fn(attempt: int):
            calls["count"] += 1
            raise LLMError(
                LLMErrorCode.CONTENT_FILTERED,
                "filtered",
                stage=LLMStage.CONTENT_REFUSAL,
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 60

        with pytest.raises(LLMError) as exc_info:
            await with_retry(fn, deadline=deadline, loop_time=loop.time)

        err = exc_info.value
        assert calls["count"] == 1
        assert err.retryable is False
        assert err.fallback_allowed is False
        assert mock_sleep.await_count == 0

    @pytest.mark.asyncio
    async def test_exponential_backoff_increases(self, mock_sleep):
        """退避时间应随 attempt 指数增长（0.5s, 1.0s, ...）。"""

        async def fn(attempt: int):
            raise LLMError(
                LLMErrorCode.NETWORK_ERROR,
                "net",
                stage=LLMStage.PRE_FIRST_TOKEN,
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 60

        with pytest.raises(LLMError):
            await with_retry(fn, deadline=deadline, loop_time=loop.time)

        sleeps = [call.args[0] for call in mock_sleep.await_args_list]
        # 第 1 次退避 ~0.5s, 第 2 次 ~1.0s（加抖动）
        assert len(sleeps) == MAX_RETRIES
        assert sleeps[0] < sleeps[1]
        assert 0.4 < sleeps[0] < 0.8
        assert 0.9 < sleeps[1] < 1.3
