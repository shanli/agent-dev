"""对比串行 vs asyncio.gather 并发抓取的执行时间。"""

import asyncio
import time

import httpx

URLS = ["https://httpbin.org/delay/1"] * 3


async def fetch_one(client: httpx.AsyncClient, url: str) -> int:
    resp = await client.get(url)
    return resp.status_code


# ── 串行：for 循环逐个 await ──
async def fetch_all_urls(urls: list[str]) -> list[int]:
    async with httpx.AsyncClient() as client:
        results = []
        for url in urls:
            results.append(await fetch_one(client, url))
        return results


# ── 并发：asyncio.gather 同时发起 ──
async def fetch_all_urls_concurrent(urls: list[str]) -> list[int]:
    async with httpx.AsyncClient() as client:
        tasks = [fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)


async def main() -> None:
    # 串行
    t0 = time.perf_counter()
    serial = await fetch_all_urls(URLS)
    serial_time = time.perf_counter() - t0
    print(f"[串行]   结果: {serial}  耗时: {serial_time:.2f}s")

    # 并发
    t0 = time.perf_counter()
    concurrent = await fetch_all_urls_concurrent(URLS)
    concurrent_time = time.perf_counter() - t0
    print(f"[并发]   结果: {concurrent}  耗时: {concurrent_time:.2f}s")

    print(f"\n加速比: {serial_time / concurrent_time:.1f}x")


if __name__ == "__main__":
    asyncio.run(main())
