from typing import Protocol

from src.schema.message import Message, ToolDefinition


class LLMProvider(Protocol):
    async def generate(self, messages: list[Message], available_tools: list[ToolDefinition]) -> Message:
        ...
