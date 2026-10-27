"""用泡茶流程演示异步 vs 同步的耗时差异。"""

import asyncio
import time


# ── 同步版本 ──
def boil_water_sync() -> None:
    time.sleep(3)
    print("  水烧开了")


def prepare_teapot_sync() -> None:
    time.sleep(1)
    print("  茶壶准备好了")


def add_tea_leaves_sync() -> None:
    time.sleep(0.5)
    print("  茶叶放好了")


def make_tea_sync() -> None:
    print("[同步] 开始泡茶...")
    boil_water_sync()
    prepare_teapot_sync()
    add_tea_leaves_sync()
    print("[同步] 茶泡好了，可以喝了")


# ── 异步版本 ──
async def boil_water() -> None:
    await asyncio.sleep(3)
    print("  水烧开了")


async def prepare_teapot() -> None:
    await asyncio.sleep(1)
    print("  茶壶准备好了")


async def add_tea_leaves() -> None:
    await asyncio.sleep(0.5)
    print("  茶叶放好了")


async def make_tea_async() -> None:
    print("[异步] 开始泡茶...")
    # 三个步骤同时进行，总耗时 = 最长的那个 = 3 秒
    await asyncio.gather(
        boil_water(),
        prepare_teapot(),
        add_tea_leaves(),
    )
    print("[异步] 茶泡好了，可以喝了")


def main() -> None:
    # 同步
    t0 = time.perf_counter()
    make_tea_sync()
    sync_time = time.perf_counter() - t0
    print(f"同步耗时: {sync_time:.2f}s\n")

    # 异步
    t0 = time.perf_counter()
    asyncio.run(make_tea_async())
    async_time = time.perf_counter() - t0
    print(f"异步耗时: {async_time:.2f}s\n")

    print(f"异步比同步快 {sync_time / async_time:.1f} 倍")


if __name__ == "__main__":
    main()
