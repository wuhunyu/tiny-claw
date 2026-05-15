from typing import Protocol

from src.schema.message import ToolCall, ToolResult, Message


class Reporter(Protocol):

    def session_start(self) -> None:
        ...

    def session_end(self) -> None:
        ...

    def step_start(self, step_count: int) -> None:
        ...

    def step_end(self, step_count: int) -> None:
        ...

    def on_thinking(self, message: Message) -> None:
        ...

    def pre_tool_call(self, tool_call: ToolCall) -> None:
        ...

    def post_tool_call(self, tool_result: ToolResult) -> None:
        ...

    def on_message(self, message: Message) -> None:
        ...
