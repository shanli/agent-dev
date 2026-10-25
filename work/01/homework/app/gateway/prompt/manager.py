"""提示词管理器：版本解析 + 模板渲染 + 变量校验 + hash 审计。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from app.core.errors import LLMError, LLMErrorCode, LLMStage
from app.core.types import PromptRef, UnifiedMessage
from app.gateway.prompt.repo import PromptRepo

_VAR_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptManager:
    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    async def resolve(self, ref: PromptRef, rec: Any) -> list[UnifiedMessage]:
        """解析提示词引用 → 渲染模板 → 返回 messages。

        rec 是调用记录对象，用于回填 prompt_name/version/hash。
        """
        async with self._session_factory() as session:
            repo = PromptRepo(session)
            pv = await repo.get_version(ref.name, ref.version)
        if pv is None:
            raise LLMError(
                LLMErrorCode.NO_COMPATIBLE_MODEL,
                f"Prompt 不存在或无激活版本: {ref.name} (version={ref.version})",
                stage=LLMStage.ROUTING,
            )

        # 变量校验：模板声明的变量必须全部提供
        declared = set(pv.variables or [])
        provided = set(ref.variables.keys())
        missing = declared - provided
        if missing:
            raise LLMError(
                LLMErrorCode.PROMPT_VARIABLE_MISSING,
                f"缺少模板变量: {sorted(missing)}",
                stage=LLMStage.PROMPT,
            )

        # 渲染模板（简单 {{var}} 替换）
        rendered = self._render(pv.template, ref.variables)

        # 回填观测字段
        rec.prompt_name = ref.name
        rec.prompt_version = pv.version
        rec.prompt_hash = hashlib.sha256(pv.template.encode("utf-8")).hexdigest()[:16]

        return self._parse_messages(rendered)

    @staticmethod
    def _render(template: str, variables: dict[str, Any]) -> str:
        def repl(m: re.Match) -> str:
            key = m.group(1)
            return str(variables.get(key, ""))

        return _VAR_PATTERN.sub(repl, template)

    @staticmethod
    def _parse_messages(rendered: str) -> list[UnifiedMessage]:
        """把渲染后的文本按角色分块解析为 messages。

        模板约定：以「role: system」「role: user」「role: assistant」行分隔不同角色。
        若没有角色标记，整体作为 user 消息。
        """
        lines = rendered.splitlines()
        messages: list[UnifiedMessage] = []
        current_role = "user"
        current_lines: list[str] = []

        for line in lines:
            m = re.match(r"^role:\s*(system|user|assistant)\s*$", line, re.IGNORECASE)
            if m:
                if current_lines:
                    messages.append(
                        UnifiedMessage(role=current_role, content="\n".join(current_lines).strip())
                    )
                current_role = m.group(1).lower()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            messages.append(
                UnifiedMessage(role=current_role, content="\n".join(current_lines).strip())
            )

        return messages or [UnifiedMessage(role="user", content=rendered)]
