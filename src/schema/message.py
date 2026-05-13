from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    ROLE_SYSTEM = "system"
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_TOOL = "tool"


class ToolCall(BaseModel):
    id: str = Field(..., description="工具调用ID")
    name: str = Field(..., description="工具名称")
    arguments: dict[str, Any] | str | None = Field(default=None, description="工具调用参数")


class ToolResult(BaseModel):
    tool_call_id: str = Field(..., description="工具调用ID")
    output: str = Field(..., description="工具调用结果")
    is_error: bool = Field(default=False, description="是否错误")


class ToolDefinition(BaseModel):
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    readonly: bool = Field(default=False, description="是否只读")
    input_schema: dict[str, Any] = Field(..., description="工具输入参数")


class Message(BaseModel):
    role: Role = Field(..., description="消息角色")
    reasoning_content: str | None = Field(default=None, description="推理内容")
    content: str | None = Field(default=None, description="消息内容")
    tool_call_id: str | None = Field(default=None, description="工具调用ID")
    tool_calls: list[ToolCall] | None = Field(default=None, description="工具调用")
