"""结构化输出校验 + 有限次自动修复。

规则（对齐阶段化决策矩阵的'输出校验'阶段）：
- 校验失败 → 把校验错误信息回传同一模型修复，默认 1 次；
- 修复预算独立于网络重试预算（MAX_RETRIES），互不侵占；
- 修复仍失败 → 抛 SCHEMA_VALIDATION_FAILED。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import jsonschema

from app.adapters.base import NormalizedRequest
from app.core.errors import LLMError, LLMErrorCode, LLMStage
from app.core.types import UnifiedRequest, UnifiedResponse

MAX_REPAIRS = 1  # 修复预算：1 次


class StructuredValidator:
    def __init__(self) -> None:
        self._validator_cls = jsonschema.Draft202012Validator

    def validate(self, content: str, schema: dict[str, Any]) -> None:
        """校验 JSON 内容是否符合 schema，失败抛异常。"""
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(
                LLMErrorCode.SCHEMA_VALIDATION_FAILED,
                f"输出不是合法 JSON: {e}",
                stage=LLMStage.OUTPUT_VALIDATION,
                cause=e,
            )
        errors = sorted(self._validator_cls(schema).iter_errors(obj), key=lambda e: e.path)
        if errors:
            msg = "; ".join(
                f"{'/'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in errors
            )
            raise LLMError(
                LLMErrorCode.SCHEMA_VALIDATION_FAILED,
                f"JSON Schema 校验失败: {msg}",
                stage=LLMStage.OUTPUT_VALIDATION,
            )

    async def validate_and_repair(
        self,
        resp: UnifiedResponse,
        req: UnifiedRequest,
        norm: NormalizedRequest,
        rec: Any,
        invoke: Callable[[NormalizedRequest, UnifiedRequest, Any, int], Awaitable[UnifiedResponse]],
    ) -> UnifiedResponse:
        """校验 → 失败则修复一次 → 仍失败上抛。

        invoke: Gateway 提供的调用封装，复用限流/计时/适配器路由。
        """
        for round_no in range(MAX_REPAIRS + 1):
            try:
                self.validate(resp.content, req.response_schema or {})
                rec.validation_result = "passed" if round_no == 0 else "repair_retried"
                return resp
            except LLMError as e:
                if round_no == MAX_REPAIRS:
                    rec.validation_result = "failed"
                    raise
                # 修复：把校验错误 + 原输出追加为新消息，让同模型重生成
                # 先把已有消息统一转成 dict，避免 UnifiedMessage(**UnifiedMessage) 解包崩溃
                repair_msgs = [m.model_dump() for m in norm.messages]
                repair_msgs.append(
                    {
                        "role": "assistant",
                        "content": resp.content,
                    }
                )
                repair_msgs.append(
                    {
                        "role": "user",
                        "content": (
                            "上一次输出未通过 JSON Schema 校验，请仅修正问题后重新输出，"
                            "不要添加任何额外解释。校验错误：\n" + e.message
                        ),
                    }
                )
                from app.core.types import UnifiedMessage

                repair_norm = norm.model_copy(
                    update={"messages": [UnifiedMessage(**m) for m in repair_msgs]}
                )
                resp = await invoke(repair_norm, req, rec, attempt=round_no + 2)
