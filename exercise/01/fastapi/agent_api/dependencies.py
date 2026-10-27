"""共享依赖：认证、数据库会话等。"""

from __future__ import annotations

from typing import Optional

from fastapi import Header

from agent_api.db import InMemoryDB, db

# 管理员密钥（演示用，实际应从环境变量读取）
ADMIN_KEY = "super-secret-admin-key-2026"


class AdminAccessDenied(Exception):
    """管理员访问被拒绝：密钥缺失或错误。"""

    def __init__(self, detail: str = "无效的管理员密钥") -> None:
        self.detail = detail
        super().__init__(detail)


def get_db() -> InMemoryDB:
    """获取数据库实例（模拟依赖注入）。"""
    return db


def verify_admin(
    key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> str:
    """校验 X-Admin-Key 请求头，通过则返回密钥，否则抛 AdminAccessDenied。"""
    if not key:
        raise AdminAccessDenied(detail="缺少 X-Admin-Key 请求头")
    if key != ADMIN_KEY:
        raise AdminAccessDenied(detail="X-Admin-Key 无效")
    return key
