from enum import Enum
from typing import Any, ClassVar

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


class OutputSchema(BaseModel):
    JSON_EXAMPLE: ClassVar[str] = """
{
  "is_final_answer": bool,
  "content": str
}
"""
    PROMPT: ClassVar[str] = f"""

你必须遵守以下输出协议：

1. 当你需要调用工具、函数或外部能力来继续完成任务时：
   - 使用系统提供的 tool_calls 发起工具调用。
   - message.content 可以为空。
   - 不要在 message.content 中描述工具调用。
   - 不要输出 JSON。

2. 当你不需要继续调用工具时，message.content 必须只输出一个合法 JSON 对象。
   JSON 必须符合以下格式：

{JSON_EXAMPLE}

字段含义：
- is_final_answer:
  - true：表示这是可以直接展示给用户的回答，包括最终回答、向用户追问必要信息、说明无法继续，或请求用户提供进一步指令。
  - false：表示当前还不能展示给用户，通常表示仍处于内部处理中，或后续还有可用工具调用可以继续推进任务。
- content:
  - 当 is_final_answer 为 true 时，content 是需要展示给用户的内容，可以是最终回答、追问、澄清、错误说明或下一步所需信息。
  - 当 is_final_answer 为 false 时，content 通常为空，或仅用于内部状态描述，不应作为面向用户的最终输出。

判断规则：
- 如果已经充分回答用户问题，并且不需要更多工具调用，设置 is_final_answer=true。
- 如果缺少完成任务所必需的信息，且无法合理默认，需要用户补充信息或给出进一步指令才能继续，设置 is_final_answer=true，并在 content 中明确说明需要用户补充什么。
- 如果工具失败、权限不足、信息不可获得，且没有可行的下一步，需要告知用户原因或请求用户进一步操作，设置 is_final_answer=true，并在 content 中说明原因。
- 如果仍然有可用工具可以帮助完成任务，优先调用工具，不要输出该 JSON。
- 只有在当前结果不应直接展示给用户，且任务仍可通过后续工具调用或内部步骤继续推进时，才设置 is_final_answer=false。
- 不要把工具调用、函数名、参数放进 content。
- 不要暴露详细思考过程。
- 不要输出 Markdown。
- 不要输出代码块。
- 不要输出 JSON 之外的任何文本。

    """

    is_final_answer: bool = Field(default=False, description="是否最终输出")
    content: str = Field(..., description="输出内容")
