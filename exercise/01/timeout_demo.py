"""演示 asyncio.wait_for 超时回退：5 秒任务设 2 秒超时，超时返回缓存默认值。"""

import asyncio


async def slow_task() -> str:
    """模拟耗时 5 秒的异步任务。"""
    await asyncio.sleep(5)
    return "任务完成，返回真实数据"


async def fetch_with_timeout(timeout: float = 2.0) -> str:
    """调用慢任务，超时后返回缓存默认值。"""
    try:
        return await asyncio.wait_for(slow_task(), timeout=timeout)
    except asyncio.TimeoutError:
        return "请求超时，使用缓存数据"


async def main() -> None:
    result = await fetch_with_timeout()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
