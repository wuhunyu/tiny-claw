import logging
from typing import Protocol, Any, TypeAlias, Callable, Awaitable

from src.core.context import Context
from src.excetion.exceptions import InvalidParamException
from src.schema.message import ToolDefinition, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class BaseTool(Protocol):
    def name(self) -> str:
        ...

    def readonly(self) -> bool:
        ...

    def definition(self) -> ToolDefinition:
        ...

    async def execute(self, context: Context, arguments: dict[str, Any] | str | None) -> str:
        ...


ToolHandler: TypeAlias = Callable[
    [Context, ToolCall],
    Awaitable[ToolResult],
]


class ToolMiddleware(Protocol):
    async def use(
            self,
            context: Context,
            tool_call: ToolCall,
            handler: ToolHandler,
    ) -> ToolResult:
        ...


class Registry(Protocol):
    async def registry(self, tool: BaseTool) -> None:
        ...

    async def get_available_tools(self) -> list[ToolDefinition]:
        ...

    async def registry_middleware(self, middleware: ToolMiddleware) -> None:
        ...

    async def execute(self, context: Context, call: ToolCall) -> ToolResult:
        ...


class RegistryImpl:
    _tools: dict[str, BaseTool]
    _middlewares: list[ToolMiddleware]
    _handler: ToolHandler

    def __init__(self):
        self._tools = {}
        self._middlewares = []
        self._handler = self._execute_tool_directly

    async def registry(self, tool: BaseTool) -> None:
        if not tool:
            raise InvalidParamException(message="工具不能为空")
        if tool.name() in self._tools:
            logger.info(f"[Registry] 工具 {tool.name()} 已存在，将要执行覆盖")
        self._tools[tool.name()] = tool

    async def get_available_tools(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    async def registry_middleware(self, middleware: ToolMiddleware) -> None:
        if not middleware:
            raise InvalidParamException(message="中间件不能为空")
        self._middlewares.append(middleware)
        self._rebuild_handler_chain()

    def _rebuild_handler_chain(self) -> None:
        handler: ToolHandler = self._execute_tool_directly

        for middleware in reversed(self._middlewares):
            next_handler = handler

            def mid_wrapped(
                    _middleware: ToolMiddleware = middleware,
                    _next_handler: ToolHandler = next_handler,
            ) -> ToolHandler:
                async def wrapped(
                        context: Context,
                        tool_call: ToolCall,
                ) -> ToolResult:
                    return await _middleware.use(
                        context=context,
                        tool_call=tool_call,
                        handler=_next_handler,
                    )

                return wrapped

            handler = mid_wrapped(
                _middleware=middleware,
                _next_handler=next_handler,
            )

        self._handler = handler

    async def execute(
            self,
            context: Context,
            call: ToolCall,
    ) -> ToolResult:
        return await self._handler(context, call)

    async def _execute_tool_directly(
            self,
            context: Context,
            call: ToolCall,
    ) -> ToolResult:
        tool = self._tools.get(call.name)

        if not tool:
            return ToolResult(
                tool_call_id=call.id,
                output=f"工具 {call.name} 不存在",
                is_error=True,
            )

        try:
            res_execute = await tool.execute(context, call.arguments)
        except:
            logger.exception(f"[Registry] 工具 {call.name} 执行失败")
            return ToolResult(
                tool_call_id=call.id,
                output=f"工具 {call.name} 执行失败",
                is_error=True,
            )

        return ToolResult(
            tool_call_id=call.id,
            output=res_execute,
            is_error=False,
        )
