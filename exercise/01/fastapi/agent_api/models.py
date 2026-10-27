"""Pydantic 数据模型：请求 / 响应 / 枚举。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── 枚举 ──

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


# ── 嵌套模型 ──

class ToolCall(BaseModel):
    id: str = Field(..., description="工具调用 ID")
    name: str = Field(..., min_length=1, max_length=64, description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")


class Message(BaseModel):
    role: MessageRole = Field(..., description="消息角色")
    content: str = Field(..., min_length=1, description="消息内容")
    tool_call_id: Optional[str] = Field(default=None, description="工具调用 ID")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="工具调用列表")

    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("消息内容不能为空")
        return stripped

    @model_validator(mode="after")
    def check_tool_call_id(self) -> "Message":
        if self.role == MessageRole.TOOL and not self.tool_call_id:
            raise ValueError("role=tool 时必须提供 tool_call_id")
        if self.role != MessageRole.TOOL and self.tool_call_id:
            raise ValueError(f"role={self.role.value} 时不应提供 tool_call_id")
        return self


class Pagination(BaseModel):
    page: int = Field(..., ge=1, description="页码")
    page_size: int = Field(..., ge=1, le=100, description="每页数量")
    total: int = Field(..., ge=0, description="总数")


class ErrorDetail(BaseModel):
    field: Optional[str] = Field(default=None, description="出错字段")
    message: str = Field(..., description="错误描述")


# ── 请求模型（extra="forbid"） ──

class CreateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128, description="Agent 名称")
    model: str = Field(..., min_length=1, max_length=128, description="模型名称")
    system_prompt: str = Field(default="你是一个有用的 AI 助手", description="系统提示词")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    max_tokens: int = Field(default=2048, ge=1, le=128000, description="最大输出 token")
    api_token: str = Field(..., min_length=1, description="API Token", exclude=True)


class UpdateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1, max_length=128, description="Agent 名称")
    model: Optional[str] = Field(default=None, min_length=1, max_length=128, description="模型名称")
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="采样温度")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=128000, description="最大输出 token")
    status: Optional[AgentStatus] = Field(default=None, description="状态")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., description="Agent ID")
    messages: list[Message] = Field(..., min_length=1, description="消息列表")
    stream: bool = Field(default=False, description="是否流式")

    @model_validator(mode="after")
    def check_first_message(self) -> "ChatRequest":
        if self.messages[0].role not in (MessageRole.USER, MessageRole.SYSTEM):
            raise ValueError("首条消息必须是 user 或 system")
        return self


# ── 响应模型 ──

class AgentResponse(BaseModel):
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="名称")
    model: str = Field(..., description="模型")
    system_prompt: str = Field(..., description="系统提示词")
    temperature: float = Field(..., description="温度")
    max_tokens: int = Field(..., description="最大 token")
    status: AgentStatus = Field(..., description="状态")
    created_at: datetime = Field(..., description="创建时间")


class AgentListResponse(BaseModel):
    items: list[AgentResponse] = Field(..., description="Agent 列表")
    pagination: Pagination = Field(..., description="分页")


class ChatResponse(BaseModel):
    agent_id: str = Field(..., description="Agent ID")
    message: Message = Field(..., description="回复消息")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="工具调用")
    usage: dict[str, int] = Field(default_factory=dict, description="token 用量")


class ErrorResponse(BaseModel):
    code: str = Field(..., description="错误码")
    message: str = Field(..., description="错误描述")
    details: list[ErrorDetail] = Field(default_factory=list, description="详情")
    request_id: Optional[str] = Field(default=None, description="请求 ID")
