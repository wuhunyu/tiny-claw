import asyncio
import logging
import os

from pydantic import ValidationError

from src.context.composer import PromptComposer
from src.engine.reporter import Reporter
from src.provider.interface import LLMProvider
from src.schema.message import Role, Message, ToolCall, ToolResult, OutputSchema
from src.tools.registry import Registry

logger = logging.getLogger(__name__)


class AgentEngine:
    provider: LLMProvider
    registry: Registry
    prompt_composer: PromptComposer
    reporter: Reporter
    work_dir: str
    enable_thinking: bool

    def __init__(
            self,
            provider: LLMProvider,
            registry: Registry,
            prompt_composer: PromptComposer,
            reporter: Reporter,
            work_dir: str = os.getenv("WORK_DIR", os.getcwd()),
            enable_thinking: bool = False,
    ):
        self.provider = provider
        self.registry = registry
        self.prompt_composer = prompt_composer
        self.reporter = reporter
        self.work_dir = work_dir
        self.enable_thinking = enable_thinking

    async def run(self, user_prompt: str):
        logger.info(f"[Engine] 引擎启动，锁定工作区: {self.work_dir}")
        logger.info(f"[Engine] 慢思考模式 (Thinking Phase): {self.enable_thinking}")

        # 加载系统提示词
        system_prompt = await self.prompt_composer.build()
        logger.info(f"[Engine] 加载系统提示词: {system_prompt}")

        context_history = [
            # 加载系统提示词
            system_prompt,
            Message(
                role=Role.ROLE_USER,
                content=user_prompt
            )
        ]

        self.reporter.session_start()
        turnCount = 0
        while True:
            turnCount += 1
            logger.info(f"========== [Turn {turnCount}] 开始 ==========")
            self.reporter.step_start(turnCount)

            if self.enable_thinking:
                logger.info(f"[Engine][Phase 1] 剥夺工具访问权，强制进入慢思考与规划阶段...")
                try:
                    think_resp, _ = self.parse_output(await self.provider.generate(context_history, []))
                    logger.info(f"[Engine][Phase 1] 思考结果: {think_resp.content}")
                    self.reporter.on_thinking(think_resp)
                except Exception as e:
                    logger.exception(f"thinking 阶段生成失败")
                    raise ValueError(f"thinking 阶段生成失败: {e}")

                if think_resp.content:
                    context_history.append(think_resp)

            available_tools = await self.registry.get_available_tools()
            try:
                logger.info(f"[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")
                action_resp, is_final_answer = self.parse_output(
                    await self.provider.generate(context_history, available_tools)
                )
            except Exception as e:
                logger.exception(f"action 阶段生成失败")
                raise ValueError(f"action 阶段生成失败: {e}")

            context_history.append(action_resp)
            if action_resp.content:
                logger.info(f"🤖 [对外回复]: {action_resp.content}")
                self.reporter.on_message(action_resp)

            if is_final_answer:
                logger.info("[Engine] 模型未请求调用工具，任务宣告完成。")
                self.reporter.step_end(turnCount)
                self.reporter.session_end()
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
                    self.reporter.pre_tool_call(tc)
                    tr = await self.registry.execute(tc)
                    self.reporter.post_tool_call(tr)
                    return tr

                tasks = [
                    asyncio.create_task(wrap_tool_execute(tool_call))
                    for tool_call in action_resp.tool_calls
                ]
                tool_results = await asyncio.gather(*tasks, return_exceptions=True)
                for tool_call, tool_result in zip(action_resp.tool_calls, tool_results):
                    if isinstance(tool_result, Exception):
                        context_history.append(Message(
                            role=Role.ROLE_TOOL,
                            tool_call_id=tool_call.id,
                            content=f"工具 {tool_call.name} 执行失败: {tool_result!r}",
                        ))
                    else:
                        context_history.append(handle_tool_result(tool_call, tool_result))
            else:
                # 串行执行
                for tool_call in action_resp.tool_calls:
                    logger.info(f"-> 🛠️ 串行执行工具: {tool_call.name}, 参数: {tool_call.arguments}")
                    self.reporter.pre_tool_call(tool_call)
                    tool_result = await self.registry.execute(tool_call)
                    self.reporter.post_tool_call(tool_result)
                    context_history.append(handle_tool_result(tool_call, tool_result))

            self.reporter.step_end(turnCount)

    def parse_output(self, message: Message) -> tuple[Message, bool]:
        message_content = message.content
        if not message_content:
            return message, False
        try:
            output = OutputSchema.model_validate_json(message_content)
        except ValidationError as e:
            logger.warning("输出结果不符合格式", exc_info=True)
            raise ValueError(f"""
输出结果不符合格式:
{OutputSchema.JSON_EXAMPLE}
""")
        message.content = output.content
        return message, output.is_final_answer
