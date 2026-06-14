import asyncio
import logging
from src.config.config import settings
from src.context.compactor import Compactor
from src.context.recovery import ToolRecoveryManager
from src.core.context import Context
from src.engine.channel import ChannelMessage
from src.engine.reporter import Reporter
from src.engine.session import Session
from src.excetion.exceptions import TinyClawException
from src.provider.interface import LLMProvider
from src.schema.message import Role, Message, ToolCall, ToolResult
from src.tools.registry import Registry

logger = logging.getLogger(__name__)


class AgentEngine:
    provider: LLMProvider
    compactor: Compactor
    registry: Registry
    reporter: Reporter
    tool_recovery_manager: ToolRecoveryManager
    work_dir: str
    enable_thinking: bool

    def __init__(
            self,
            provider: LLMProvider,
            compactor: Compactor,
            registry: Registry,
            reporter: Reporter,
            channel_message: ChannelMessage,
            tool_recovery_manager: ToolRecoveryManager,
            work_dir: str = settings.work_dir,
            enable_thinking: bool = False,
    ):
        self.provider = provider
        self.compactor = compactor
        self.registry = registry
        self.reporter = reporter
        self.channel_message = channel_message
        self.tool_recovery_manager = tool_recovery_manager
        self.work_dir = work_dir
        self.enable_thinking = enable_thinking

    async def run(
            self,
            context: Context,
            user_prompt: str,
            system_prompt: Message,
            session: Session,
    ):
        logger.info(f"[Engine] 引擎启动，锁定工作区: {self.work_dir}")
        logger.info(f"[Engine] 慢思考模式 (Thinking Phase): {self.enable_thinking}")

        initial_messages = [
            # 加载系统提示词
            system_prompt,
        ]
        user_prompt = Message(
            role=Role.ROLE_USER,
            content=user_prompt
        )
        if await session.is_inited():
            # 加载系统提示词
            logger.info(f"[Engine] 加载系统提示词: {system_prompt}")
            await session.append(initial_messages + [user_prompt])
        else:
            await session.append(user_prompt)

        self.reporter.session_start(context)
        turnCount = 0
        while True:
            turnCount += 1
            logger.info(f"========== [Turn {turnCount}] 开始 ==========")
            self.reporter.step_start(context, turnCount)

            if self.enable_thinking:
                logger.info(f"[Engine][Phase 1] 剥夺工具访问权，强制进入慢思考与规划阶段...")
                try:
                    context_history = await self.compact(context, session, initial_messages)
                    think_resp, _ = await self.provider.generate(context, context_history, [])
                    logger.info(f"[Engine][Phase 1] 思考结果: {think_resp.content}")
                    self.reporter.on_thinking(context, think_resp)
                except Exception as e:
                    logger.exception(f"thinking 阶段生成失败")
                    raise TinyClawException(message=f"thinking 阶段生成失败") from e

                if think_resp.content:
                    await session.append([think_resp])

            available_tools = await self.registry.get_available_tools()
            try:
                logger.info(f"[Engine][Phase 2] 恢复工具挂载: {[tool.name for tool in available_tools]}")
                context_history = await self.compact(context, session, initial_messages)
                action_resp, _ = await self.provider.generate(context, context_history, available_tools)
            except Exception as e:
                logger.exception(f"action 阶段生成失败")
                raise TinyClawException(message=f"action 阶段生成失败") from e

            await session.append(action_resp)
            if action_resp.content:
                logger.info(f"🤖 [对外回复]: {action_resp.content}")
                self.reporter.on_message(context, action_resp)
                await session.append(action_resp)
                self.reporter.step_end(context, turnCount)
                self.reporter.session_end(context)
                logger.info("[Engine] 模型未请求调用工具，任务宣告完成。")
                break

            logger.info(f"[Engine] 模型请求调用 {len(action_resp.tool_calls)} 个工具...")

            # 判断本次所有的 tool 是否为 readonly
            available_tool_dict = {tool.name: tool for tool in available_tools}
            is_all_readonly = True
            for tool_call in action_resp.tool_calls:
                tool = available_tool_dict.get(tool_call.name, None)
                if not tool or not tool.readonly:
                    is_all_readonly = False
                    break

            def handle_tool_result(tc: ToolCall, tr: ToolResult) -> Message:
                if not tr.is_error:
                    logger.info(f"-> ✅ 工具执行成功 (返回 {len(tr.output)} 字节)")
                else:
                    logger.info(f"-> ❌ 工具执行失败: {tr.output}")
                return Message(
                    role=Role.ROLE_TOOL,
                    tool_call_id=tc.id,
                    content=tr.output,
                )

            if is_all_readonly:
                # 并发读
                async def wrap_tool_execute(tc: ToolCall) -> ToolResult:
                    logger.info(f"-> 🛠️ 并发执行工具: {tc.name}, 参数: {tc.arguments}")
                    self.reporter.pre_tool_call(context, tc)
                    tr = await self.registry.execute(context, tc)
                    self.reporter.post_tool_call(context, tr)
                    return tr

                tasks = [
                    asyncio.create_task(wrap_tool_execute(tool_call))
                    for tool_call in action_resp.tool_calls
                ]
                tool_results = await asyncio.gather(*tasks, return_exceptions=True)
                for tool_call, tool_result in zip(action_resp.tool_calls, tool_results):
                    if isinstance(tool_result, Exception):
                        await session.append(Message(
                            role=Role.ROLE_TOOL,
                            tool_call_id=tool_call.id,
                            content=f"工具 {tool_call.name} 执行失败: {tool_result!r}",
                        ))
                    else:
                        if tool_result.is_error:
                            suggest_content = await self.tool_recovery_manager.analyze(
                                context,
                                tool_call,
                                tool_result.output,
                            )
                            await session.append(
                                Message(
                                    role=Role.ROLE_TOOL,
                                    tool_call_id=tool_call.id,
                                    content=suggest_content,
                                )
                            )
                        else:
                            await session.append(handle_tool_result(tool_call, tool_result))
            else:
                # 串行执行
                for tool_call in action_resp.tool_calls:
                    logger.info(f"-> 🛠️ 串行执行工具: {tool_call.name}, 参数: {tool_call.arguments}")
                    self.reporter.pre_tool_call(context, tool_call)
                    tool_result = await self.registry.execute(context, tool_call)
                    self.reporter.post_tool_call(context, tool_result)
                    if tool_result.is_error:
                        suggest_content = await self.tool_recovery_manager.analyze(
                            context,
                            tool_call,
                            tool_result.output,
                        )
                        await session.append(
                            Message(
                                role=Role.ROLE_TOOL,
                                tool_call_id=tool_call.id,
                                content=suggest_content,
                            )
                        )
                    else:
                        await session.append(handle_tool_result(tool_call, tool_result))

            self.reporter.step_end(context, turnCount)

    async def compact(self, context: Context, session: Session, initial_messages: list[Message]) -> list[Message]:
        return self.compactor.compact(
            context=context,
            messages=initial_messages + await session.get_working_memory(settings.max_session_window_size),
        )
