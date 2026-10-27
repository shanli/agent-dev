"""AI Agent 服务的 Pydantic API 数据模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ─────────────── 枚举 ───────────────

class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ─────────────── 嵌套模型 ───────────────

class ToolCall(BaseModel):
    """模型发起的工具调用。"""

    id: str = Field(..., description="工具调用唯一 ID")
    name: str = Field(..., description="工具名称", min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具调用参数")


class Message(BaseModel):
    """对话消息。"""

    role: MessageRole = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容", min_length=1)
    tool_call_id: Optional[str] = Field(default=None, description="工具调用 ID（仅 role=tool 时使用）")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="assistant 发起的工具调用列表")

    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        """去除首尾空白，避免空字符串绕过 min_length。"""
        stripped = v.strip()
        if not stripped:
            raise ValueError("消息内容不能为空")
        return stripped

    @model_validator(mode="after")
    def check_tool_call_id(self) -> "Message":
        """role=tool 时必须提供 tool_call_id；其他角色不应有。"""
        if self.role == MessageRole.TOOL and not self.tool_call_id:
            raise ValueError("role=tool 时必须提供 tool_call_id")
        if self.role != MessageRole.TOOL and self.tool_call_id:
            raise ValueError(f"role={self.role.value} 时不应提供 tool_call_id")
        return self


class Pagination(BaseModel):
    """分页信息。"""

    page: int = Field(..., ge=1, description="当前页码（从 1 开始）")
    page_size: int = Field(..., ge=1, le=100, description="每页数量（1-100）")
    total: int = Field(..., ge=0, description="总记录数")


class ErrorDetail(BaseModel):
    """错误详情。"""

    field: Optional[str] = Field(default=None, description="出错字段（校验错误时）")
    message: str = Field(..., description="错误描述")


# ─────────────── 请求模型（extra="forbid"） ───────────────

class CreateAgentRequest(BaseModel):
    """创建 Agent 请求。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128, description="Agent 名称")
    model: str = Field(..., min_length=1, max_length=128, description="使用的模型名称")
    system_prompt: str = Field(default="你是一个有用的 AI 助手", description="系统提示词")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度（0-2）")
    max_tokens: int = Field(default=2048, ge=1, le=128000, description="最大输出 token 数")
    api_token: str = Field(
        ...,
        min_length=1,
        description="上游模型 API Token（序列化时排除，避免泄露）",
        exclude=True,
    )


class ChatRequest(BaseModel):
    """对话请求。"""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., description="Agent ID")
    messages: list[Message] = Field(..., min_length=1, description="对话消息列表")
    stream: bool = Field(default=False, description="是否流式输出")

    @model_validator(mode="after")
    def check_first_message(self) -> "ChatRequest":
        """首条消息必须是 user 或 system。"""
        first = self.messages[0]
        if first.role not in (MessageRole.USER, MessageRole.SYSTEM):
            raise ValueError("对话首条消息必须是 user 或 system 角色")
        return self


class ToolExecuteRequest(BaseModel):
    """手动触发工具执行请求。"""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., description="Agent ID")
    tool_name: str = Field(..., min_length=1, max_length=64, description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")


# ─────────────── 响应模型 ───────────────

class AgentResponse(BaseModel):
    """Agent 详情响应。"""

    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent 名称")
    model: str = Field(..., description="模型名称")
    system_prompt: str = Field(..., description="系统提示词")
    temperature: float = Field(..., description="采样温度")
    max_tokens: int = Field(..., description="最大输出 token 数")
    status: AgentStatus = Field(..., description="当前状态")
    created_at: datetime = Field(..., description="创建时间")


class ChatResponse(BaseModel):
    """对话响应。"""

    agent_id: str = Field(..., description="Agent ID")
    message: Message = Field(..., description="assistant 回复的消息")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="模型发起的工具调用列表")
    usage: dict[str, int] = Field(default_factory=dict, description="token 用量")


class ToolExecuteResponse(BaseModel):
    """工具执行结果响应。"""

    tool_name: str = Field(..., description="工具名称")
    success: bool = Field(..., description="是否执行成功")
    result: Any = Field(default=None, description="执行结果")
    error: Optional[str] = Field(default=None, description="错误信息（失败时）")


class AgentListResponse(BaseModel):
    """Agent 列表响应（分页）。"""

    items: list[AgentResponse] = Field(..., description="Agent 列表")
    pagination: Pagination = Field(..., description="分页信息")


class ErrorResponse(BaseModel):
    """统一错误响应。"""

    code: str = Field(..., description="错误码")
    message: str = Field(..., description="错误描述")
    details: list[ErrorDetail] = Field(default_factory=list, description="错误详情列表")
    request_id: Optional[str] = Field(default=None, description="请求 ID（用于排查）")
