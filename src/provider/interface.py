from enum import Enum
from typing import Protocol

from src.core.context import Context
from src.schema.message import Message, ToolDefinition, TokenUsage


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMProvider(Protocol):
    async def generate(
            self,
            context: Context,
            messages: list[Message],
            available_tools: list[ToolDefinition],
    ) -> tuple[
        Message,
        TokenUsage,
    ]:
        ...
