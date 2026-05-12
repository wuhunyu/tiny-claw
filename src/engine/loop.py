import os

from src.provider.interface import LLMProvider
from src.schema.message import Role, Message
from src.tools.registry import Registry


class AgentEngine:
    provider: LLMProvider
    registry: Registry
    work_dir: str
    enable_thinking: bool

    def __init__(
            self,
            provider: LLMProvider,
            registry: Registry,
            work_dir: str = os.getenv("WORK_DIR", os.getcwd()),
            enable_thinking: bool = False,
    ):
        self.provider = provider
        self.registry = registry
        self.work_dir = work_dir
        self.enable_thinking = enable_thinking

    async def run(self, user_prompt: str):
        print(f"[Engine] 引擎启动，锁定工作区: {self.work_dir}")
        print(f"[Engine] 慢思考模式 (Thinking Phase): {self.enable_thinking}")

        context_history = [
            Message(
                role=Role.ROLE_SYSTEM,
                content="You are tiny-claw, an expert coding assistant. You have full access to tools in the workspace."
            ),
            Message(
                role=Role.ROLE_USER,
                content=user_prompt
            )
        ]

        turnCount = 0
        while True:
            turnCount += 1
            print(f"========== [Turn {turnCount}] 开始 ==========")

            if self.enable_thinking:
                print(f"[Engine][Phase 1] 剥夺工具访问权，强制进入慢思考与规划阶段...")
                try:
                    think_resp = await self.provider.generate(context_history, [])
                except Exception as e:
                    raise ValueError(f"thinking 阶段生成失败: {e}")

                if think_resp.content:
                    context_history.append(think_resp)

            available_tools = await self.registry.get_available_tools()
            try:
                print(f"[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")
                action_resp = await self.provider.generate(context_history, available_tools)
            except Exception as e:
                raise ValueError(f"action 阶段生成失败: {e}")

            context_history.append(action_resp)
            if action_resp.content:
                print(f"🤖 [对外回复]: {action_resp.content}")

            if not action_resp.tool_calls:
                print("[Engine] 模型未请求调用工具，任务宣告完成。")
                break

            print(f"[Engine] 模型请求调用 {len(action_resp.tool_calls)} 个工具...")
            for tool_call in action_resp.tool_calls:
                print(f"-> 🛠️ 执行工具: {tool_call.name}, 参数: {tool_call.arguments}")
                tool_result = await self.registry.execute(tool_call)
                if not tool_result.is_error:
                    print(f"-> ✅ 工具执行成功 (返回 {len(tool_result.output)} 字节)")
                else:
                    print(f"-> ❌ 工具执行失败: {tool_result.output}")
                context_history.append(Message(
                    role=Role.ROLE_TOOL,
                    tool_call_id=tool_call.id,
                    content=tool_result.output,
                ))
