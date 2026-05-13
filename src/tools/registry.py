import logging
from typing import Protocol, Any

from src.schema.message import ToolDefinition, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class BaseTool(Protocol):
    def name(self) -> str:
        ...

    def readonly(self) -> bool:
        ...

    def definition(self) -> ToolDefinition:
        ...

    async def execute(self, arguments: dict[str, Any] | str | None) -> str:
        ...


class Registry(Protocol):
    async def registry(self, tool: BaseTool) -> None:
        ...

    async def get_available_tools(self) -> list[ToolDefinition]:
        ...

    async def execute(self, call: ToolCall) -> ToolResult:
        ...


class RegistryImpl:
    _tools: dict[str, BaseTool] = {}

    async def registry(self, tool: BaseTool) -> None:
        if not tool:
            raise ValueError("工具不能为空")
        if tool.name() in self._tools:
            logger.info(f"[Registry] 工具 {tool.name()} 已存在，将要执行覆盖")
        self._tools[tool.name()] = tool

    async def get_available_tools(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name, None)
        if not tool:
            return ToolResult(
                tool_call_id=call.id,
                output=f"工具 {call.name} 不存在",
                is_error=True,
            )

        try:
            res_execute = await tool.execute(call.arguments)
            return ToolResult(
                tool_call_id=call.id,
                output=res_execute,
                is_error=False,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                output=f"工具 {call.name} 执行失败: {e}",
                is_error=True,
            )
