"""API Key 认证系统演示：Admin Key 鉴权 + 自定义异常处理。"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse

# ── 管理员密钥（演示用，实际应从环境变量读取） ──
ADMIN_KEY = "super-secret-admin-key-2026"


# ── 自定义异常 ──

class AdminAccessDenied(Exception):
    """管理员访问被拒绝：密钥缺失或错误。"""

    def __init__(self, detail: str = "无效的管理员密钥") -> None:
        self.detail = detail
        super().__init__(detail)


# ── 认证依赖 ──

def verify_admin(key: Optional[str] = Header(default=None, alias="X-Admin-Key")) -> str:
    """校验 X-Admin-Key 请求头，通过则返回密钥，否则抛 AdminAccessDenied。"""
    if not key:
        raise AdminAccessDenied(detail="缺少 X-Admin-Key 请求头")
    if key != ADMIN_KEY:
        raise AdminAccessDenied(detail="X-Admin-Key 无效")
    return key


# ── FastAPI 应用 ──

app = FastAPI(title="API Key 认证演示", version="0.1.0")


# 全局异常处理器：捕获 AdminAccessDenied，返回自定义格式
@app.exception_handler(AdminAccessDenied)
async def admin_access_denied_handler(request: Request, exc: AdminAccessDenied) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": "ADMIN_ACCESS_DENIED",
            "detail": exc.detail,
        },
    )


@app.get("/public")
def public() -> dict:
    """公开接口，无需认证。"""
    return {"message": "这是公开内容，任何人都能访问"}


@app.get("/admin")
def admin(key: str = Depends(verify_admin)) -> dict:
    """管理员接口，必须提供正确的 X-Admin-Key。"""
    return {"message": "欢迎管理员！你已通过认证", "key_preview": f"{key[:6]}..."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
