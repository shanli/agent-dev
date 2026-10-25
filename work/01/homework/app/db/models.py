"""数据模型：九维可观测字段 + Prompt 版本管理。

字段维度对齐'八、Token、Cost、Latency 与 Trace'：
关联 / 路由 / Prompt / 用量 / 延迟 / 弹性 / 结果 / 错误 / 成本
"""

from __future__ import annotations

from typing import Optional

import secrets
from datetime import datetime
from decimal import Decimal

from sqlalchemy import JSON, DateTime, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

Base = declarative_base()


def _id() -> str:
    return secrets.token_hex(16)


class LLMCallRecord(Base):
    """一次 attempt 的完整观测记录。同一次逻辑调用共享 call_id。"""

    __tablename__ = "llm_call_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)

    # ── 关联 ──
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    step_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    call_id: Mapped[str] = mapped_column(String(64), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    is_final: Mapped[bool] = mapped_column(default=False)

    # ── 路由 ──
    logical_model: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32))
    actual_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    candidates: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    reject_reason: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # ── Prompt ──
    prompt_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prompt_hash: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    schema_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # ── 请求特征 ──
    stream: Mapped[bool] = mapped_column(default=False)
    structured: Mapped[bool] = mapped_column(default=False)

    # ── 用量 ──
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # ── 延迟 (ms) ──
    queue_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    route_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ttft_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    generation_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # ── 弹性 ──
    fallback_from: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    timeout_budget_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # ── 结果 ──
    final_state: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # success/error/timeout/cancelled
    finish_reason: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    validation_result: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # passed/repair_retried/failed

    # ── 错误 ──
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    http_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    provider_request_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # ── 成本（写入时估算）──
    cost_input: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 6), nullable=True)
    cost_output: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 6), nullable=True)
    cost_cached: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 6), nullable=True)
    cost_total: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 6), nullable=True)
    currency: Mapped[str] = mapped_column(String(8), default="USD")

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_model_time", "logical_model", "started_at"),
        Index("ix_err_time", "error_code", "started_at"),
    )


class PromptVersion(Base):
    """提示词版本：版本不可变，回滚 = 切换 is_active。"""

    __tablename__ = "prompt_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer)
    template: Mapped[str] = mapped_column(Text)
    variables: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # 声明的变量名列表
    is_active: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_name_version", "name", "version", unique=True),)
