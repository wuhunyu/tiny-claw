import json
import logging
import re
from asyncio import Lock

from src.core.context import Context
from src.engine.channel import ChannelMessage
from src.schema.message import ToolCall, ToolResult
from src.tools.bash import Bash
from src.tools.registry import ToolHandler

logger = logging.getLogger(__name__)


class ApprovalManager:
    DANGEROUS_PATTERNS = [
        r"rm\s+-r",
        r"sudo\s+",
        r"drop\s+",
        r">.*\.go",
    ]
    APPROVE = "approve"
    REJECT = "reject"

    def __init__(self, channel_message: ChannelMessage):
        self.channel_message = channel_message
        self._lock = Lock()

    async def use(
            self,
            context: Context,
            tool_call: ToolCall,
            handler: ToolHandler,
    ) -> ToolResult:
        # 只拦截 bash
        if tool_call.name != Bash.name():
            return await handler(context, tool_call)

        # 不是危险指令
        if not ApprovalManager._is_dangerous_bash_command(tool_call.arguments.get("command", [])):
            return await handler(context, tool_call)

        with self._lock:
            logger.info(
                "⚠️ 高危操作审批请求, 工具: %s, 参数: %s",
                tool_call.name,
                json.dumps(tool_call.arguments, ensure_ascii=False),
            )

            question = f"""⚠️ **高危操作审批请求**
tiny claw 试图执行以下动作:
- 工具: {tool_call.name}
- 参数: {json.dumps(tool_call.arguments, ensure_ascii=False)}
👉 请在此消息下方回复 "{ApprovalManager.APPROVE}" 或 "{ApprovalManager.REJECT}" 来决定是否放行。"""

            # 等待审核恢回复
            answer = await self.channel_message.send_and_receive(
                context,
                question,
            )
        if answer and answer.lower() == ApprovalManager.APPROVE:
            logger.info(
                "审核通过: 工具: %s, 参数: %s",
                tool_call.name,
                json.dumps(tool_call.arguments, ensure_ascii=False),
            )
            return await handler(context, tool_call)

        logger.warning(
            "审核拒绝: 工具: %s, 参数: %s",
            tool_call.name,
            json.dumps(tool_call.arguments, ensure_ascii=False),
        )
        return ToolResult(
            tool_call_id=tool_call.id,
            output="管理员判定该操作存在极高风险，已被阻止执行",
            is_error=True,
        )

    @staticmethod
    def _is_dangerous_bash_command(commands: list[str] | None) -> bool:
        if not commands:
            return False
        command = " ".join(commands)
        for pattern in ApprovalManager.DANGEROUS_PATTERNS:
            if re.match(pattern, command):
                return True
        return False
