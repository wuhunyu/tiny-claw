import logging
import os
from enum import Enum

from litellm import acompletion
from pydantic import BaseModel, Field

from src.schema.message import Message, ToolDefinition, Role, ToolCall

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class MyChat(BaseModel):
    llm_provider: Provider = Field(default=Provider.OPENAI, description="模型供应商")
    base_url: str = Field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""), description="模型供应商的 API 地址")
    api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""), description="模型供应商的 API Key")
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "GLM-4.7-Flash"), description="模型名称")

    async def generate(self, messages: list[Message], available_tools: list[ToolDefinition]) -> Message:
        # 消息转换
        openai_messages = []
        for message in messages:
            if message.role == Role.ROLE_SYSTEM:
                openai_messages.append({"role": "system", "content": message.content})
            elif message.role == Role.ROLE_TOOL:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content
                })
            elif message.role == Role.ROLE_USER:
                openai_messages.append({"role": "user", "content": message.content})
            elif message.role == Role.ROLE_ASSISTANT:
                tool_calls = []
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_calls.append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments
                            }
                        })
                    openai_messages.append({
                        "role": "assistant",
                        "reasoning_content": message.reasoning_content,
                        "content": message.content,
                        "tool_calls": tool_calls,
                    })
                else:
                    openai_messages.append({
                        "role": "assistant",
                        "reasoning_content": message.reasoning_content,
                        "content": message.content,
                    })
            else:
                raise ValueError(f"Invalid role: {message.role}")

        # 工具定义转换
        tool_defines = []
        for available_tool in available_tools:
            tool_defines.append({
                "type": "function",
                "function": {
                    "name": available_tool.name,
                    "description": available_tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": available_tool.input_schema.get("properties", {}),
                        "required": available_tool.input_schema.get("required", []),
                        "additionalProperties": False
                    },
                    "strict": True
                }
            })

        try:
            openai_res = await acompletion(
                base_url=self.base_url,
                api_key=self.api_key,
                model=f"{self.llm_provider.value}/{self.model}",
                messages=openai_messages,
                tools=tool_defines,
            )
        except Exception as e:
            logger.exception(f"{self.llm_provider.value} API 响应失败")
            raise ValueError(f"{self.llm_provider.value} API 请求失败", e)

        if not openai_res.choices:
            logger.exception(f"{self.llm_provider.value} API 响应结果为空")
            raise ValueError(f"{self.llm_provider.value} API 响应结果为空")

        res_message = openai_res.choices[0].message
        res_tool_calls = []
        if res_message.tool_calls:
            for tool_call in res_message.tool_calls:
                res_tool_calls.append(ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments
                ))
        return Message(
            role=Role.ROLE_ASSISTANT,
            reasoning_content=getattr(res_message, "reasoning_content", None),
            content=res_message.content,
            tool_calls=res_tool_calls
        )
