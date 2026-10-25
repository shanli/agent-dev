"""SSE 解析：从响应流中逐行提取 data 事件。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[dict]:
    """逐行解析 text/event-stream，产出 data 字段的 JSON 对象。"""
    async for line in response.aiter_lines():
        if not line:
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue
        # event: / 其他行忽略
