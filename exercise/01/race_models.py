"""race_models: 同时调用多个模型，取最快返回的结果，取消其余调用。"""

import asyncio
import contextlib
import time


async def model_a(prompt: str) -> str:
    """模拟模型 A，耗时 3 秒。"""
    await asyncio.sleep(3)
    return f"[模型A] {prompt} -> A 的回答"


async def model_b(prompt: str) -> str:
    """模拟模型 B，耗时 1 秒。"""
    await asyncio.sleep(1)
    return f"[模型B] {prompt} -> B 的回答"


async def model_c(prompt: str) -> str:
    """模拟模型 C，耗时 2 秒。"""
    await asyncio.sleep(2)
    return f"[模型C] {prompt} -> C 的回答"


async def race_models(prompt: str) -> str:
    """同时调用 3 个模型，返回最先完成的结果，取消其余。"""
    tasks = {
        asyncio.create_task(model_a(prompt)): "A",
        asyncio.create_task(model_b(prompt)): "B",
        asyncio.create_task(model_c(prompt)): "C",
    }

    # 等待任意一个完成
    done, pending = await asyncio.wait(
        tasks.keys(),
        return_when=asyncio.FIRST_COMPLETED,
    )

    # 取消还在运行的任务
    for task in pending:
        task.cancel()
    # 等被取消的任务真正结束（抑制 CancelledError）
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*pending, return_exceptions=True)

    # 取最快完成的结果
    winner = done.pop()
    model_name = tasks[winner]
    result = winner.result()
    print(f"  获胜模型: {model_name}")
    return result


async def main() -> None:
    prompt = "你好"
    print(f"开始竞速，prompt: {prompt}")
    t0 = time.perf_counter()
    result = await race_models(prompt)
    elapsed = time.perf_counter() - t0
    print(f"结果: {result}")
    print(f"总耗时: {elapsed:.2f}s（最快模型 B 为 1s）")


if __name__ == "__main__":
    asyncio.run(main())
