"""FastAPI 入口：创建 app、挂载路由、配置中间件。"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent_api.dependencies import AdminAccessDenied
from agent_api.routers.agents import router as agents_router

app = FastAPI(
    title="Agent CRUD API",
    version="0.1.0",
    description="AI Agent 服务的增删改查接口",
)

# ── 操作日志 logger（确保有 handler 可见输出） ──
_ops_logger = logging.getLogger("agent_api.operations")
if not _ops_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _ops_logger.addHandler(_handler)
_ops_logger.setLevel(logging.INFO)
_ops_logger.propagate = False

# ── 中间件 ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 频率限制：同一 IP 每 10 秒最多 5 次请求 ──
RATE_LIMIT_WINDOW = 10  # 滑动窗口大小（秒）
RATE_LIMIT_MAX = 5      # 窗口内最大请求数
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = Lock()


def _get_client_ip(request: Request) -> str:
    """提取客户端 IP，优先读取 X-Forwarded-For 首部（反代场景）。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # 取最左侧的原始客户端 IP
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """全局限流：同一 IP 在滑动窗口内最多 RATE_LIMIT_MAX 次请求。"""
    ip = _get_client_ip(request)
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    with _rate_lock:
        bucket = _rate_buckets[ip]
        # 清理窗口外的时间戳
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_MAX:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "detail": (
                        f"请求过于频繁：同一 IP 每 {RATE_LIMIT_WINDOW} 秒"
                        f"最多 {RATE_LIMIT_MAX} 次请求"
                    ),
                },
            )
        bucket.append(now)

    return await call_next(request)


@app.middleware("http")
async def operation_log_middleware(request: Request, call_next):
    """记录创建(POST)/更新(PUT)操作：操作人、操作类型、时间、结果。"""
    response = await call_next(request)

    path = request.url.path
    method = request.method
    # 仅记录 /api/v1/agents 下的 POST（创建）与 PUT（更新）
    if path.startswith("/api/v1/agents") and method in ("POST", "PUT"):
        operator = request.headers.get("x-user", "anonymous")
        op_type = "create" if method == "POST" else "update"
        _ops_logger.info(
            "op_type=%s operator=%s status=%s time=%s path=%s",
            op_type,
            operator,
            response.status_code,
            datetime.now(timezone.utc).isoformat(),
            path,
        )

    return response


# ── 全局异常处理器：AdminAccessDenied → 自定义 401 格式 ──
@app.exception_handler(AdminAccessDenied)
async def admin_access_denied_handler(request: Request, exc: AdminAccessDenied) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"error": "ADMIN_ACCESS_DENIED", "detail": exc.detail},
    )


# ── 路由 ──
app.include_router(agents_router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "Agent CRUD API",
        "docs": "/docs",
        "endpoints": [
            "POST   /api/v1/agents",
            "GET    /api/v1/agents",
            "GET    /api/v1/agents/search?q=xxx",
            "GET    /api/v1/agents/{agent_id}",
            "PUT    /api/v1/agents/{agent_id}",
            "DELETE /api/v1/agents/{agent_id}",
        ],
    }
