from typing import Protocol

from src.core.context import Context
from src.schema.message import ToolCall, ToolResult, Message


class Reporter(Protocol):

    def session_start(self, context: Context) -> None:
        ...

    def session_end(self, context: Context) -> None:
        ...

    def step_start(self, context: Context, step_count: int) -> None:
        ...

    def step_end(self, context: Context, step_count: int) -> None:
        ...

    def on_thinking(self, context: Context, message: Message) -> None:
        ...

    def pre_tool_call(self, context: Context, tool_call: ToolCall) -> None:
        ...

    def post_tool_call(self, context: Context, tool_result: ToolResult) -> None:
        ...

    def on_message(self, context: Context, message: Message) -> None:
        ...
