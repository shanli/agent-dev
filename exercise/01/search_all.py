"""search_all: 并发搜索多个源，成功 >= 2 个则合并返回，否则报错。"""

import asyncio
import time


async def search_docs(query: str) -> list[str]:
    """文档库，耗时 0.8s。"""
    await asyncio.sleep(0.8)
    return [f"docs:{query}#1", f"docs:{query}#2"]


async def search_code(query: str) -> list[str]:
    """代码库，耗时 1.2s。"""
    await asyncio.sleep(1.2)
    return [f"code:{query}#1"]


async def search_web(query: str) -> list[str]:
    """网络搜索，耗时 1.5s。"""
    await asyncio.sleep(1.5)
    return [f"web:{query}#1", f"web:{query}#2", f"web:{query}#3"]


async def search_cache(query: str) -> list[str]:
    """缓存，耗时 0.3s。"""
    await asyncio.sleep(0.3)
    return [f"cache:{query}#1"]


# 注册源：名称 -> 协程函数
SOURCES = {
    "docs": search_docs,
    "code": search_code,
    "web": search_web,
    "cache": search_cache,
}


async def search_all(query: str, min_success: int = 2) -> list[str]:
    """并发搜索所有源，成功数 >= min_success 时合并返回，否则抛错。"""
    names = list(SOURCES.keys())
    coros = [SOURCES[name](query) for name in names]

    # return_exceptions=True：失败以异常对象返回，不中断其他任务
    results = await asyncio.gather(*coros, return_exceptions=True)

    merged: list[str] = []
    failed: list[str] = []
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            failed.append(name)
        else:
            merged.extend(result)

    success_count = len(names) - len(failed)
    print(f"  成功: {success_count}/{len(names)}, 失败源: {failed}")

    if success_count >= min_success:
        return merged
    raise RuntimeError(
        f"搜索失败：仅 {success_count} 个源成功（需要 >= {min_success}），失败: {failed}"
    )


async def main() -> None:
    query = "asyncio"

    print(f"[场景1] 全部成功")
    t0 = time.perf_counter()
    result = await search_all(query)
    print(f"  合并结果 ({len(result)} 条): {result}")
    print(f"  耗时: {time.perf_counter() - t0:.2f}s\n")

    print(f"[场景2] 模拟部分失败（docs 和 web 抛错）")
    # 临时把两个源改成抛异常
    original_docs, original_web = SOURCES["docs"], SOURCES["web"]

    async def fail_docs(q: str) -> list[str]:
        await asyncio.sleep(0.8)
        raise ConnectionError("文档库连接超时")

    async def fail_web(q: str) -> list[str]:
        await asyncio.sleep(1.5)
        raise TimeoutError("网络搜索超时")

    SOURCES["docs"] = fail_docs
    SOURCES["web"] = fail_web
    try:
        t0 = time.perf_counter()
        result = await search_all(query)
        print(f"  合并结果 ({len(result)} 条): {result}")
        print(f"  耗时: {time.perf_counter() - t0:.2f}s")
    except RuntimeError as e:
        print(f"  报错: {e}")
    finally:
        SOURCES["docs"] = original_docs
        SOURCES["web"] = original_web


if __name__ == "__main__":
    asyncio.run(main())
