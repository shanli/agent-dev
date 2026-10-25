"""重试执行器。

语义：MAX_RETRIES=2 → 首次请求 + 最多 2 次重试 = 共 3 次尝试。
规则（对齐阶段化决策矩阵）：
- 只重试 retryable=True 的错误；
- 退避 = max(上游 Retry-After, 指数退避 0.5s×2^(n-1) + 抖动)；
- 重试前检查截止时间预算，不足则放弃（不烧调用）；
- 流式：仅在 WAITING 期（首 token 之前）可重试，mid_stream 错误直接上抛。
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.core.errors import LLMError, LLMErrorCode, LLMStage

T = TypeVar("T")
logger = logging.getLogger(__name__)

MAX_RETRIES = 2  # 首次 + 2 次重试 = 共 3 次尝试
BACKOFF_BASE_S = 0.5
JITTER_MS = 250
MIN_VIABLE_BUDGET_S = 1.0  # 剩余预算不足此值时不再发起/重试


async def with_retry(
    fn: Callable[[int], Awaitable[T]],
    *,
    deadline: float,
    loop_time: Callable[[], float],
) -> T:
    """带截止时间预算的指数退避重试。

    Args:
        fn: 接收 attempt(从 1 开始) 返回协程的可调用对象。
        deadline: 截止时间戳（秒），由调用方按 timeout_ms 计算。
        loop_time: 获取当前事件循环时间的函数（loop.time）。
    """
    for attempt in range(1, MAX_RETRIES + 2):
        remaining = deadline - loop_time()
        if remaining < MIN_VIABLE_BUDGET_S:
            raise LLMError(
                LLMErrorCode.DEADLINE_EXCEEDED,
                f"截止时间预算不足，剩余 {remaining:.2f}s",
                stage=LLMStage.QUEUE,
            )
        try:
            return await fn(attempt)
        except LLMError as e:
            if not e.retryable or attempt > MAX_RETRIES:
                raise
            # 退避：尊重上游 Retry-After，否则指数退避 + 抖动
            backoff_ms = BACKOFF_BASE_S * (2 ** (attempt - 1)) * 1000 + random.uniform(0, JITTER_MS)
            delay_ms = max(e.retry_after_ms or 0, backoff_ms)
            # 明知会超时就不再烧一次调用
            if delay_ms / 1000 >= remaining:
                raise
            logger.warning(
                "llm.retry attempt=%s code=%s stage=%s delay_ms=%.0f",
                attempt,
                e.code,
                e.stage,
                delay_ms,
            )
            await asyncio.sleep(delay_ms / 1000)
    # 理论上不会到这里（attempt > MAX_RETRIES 已 raise）
    raise LLMError(LLMErrorCode.UNKNOWN, "重试耗尽", stage=LLMStage.PRE_FIRST_TOKEN)
