import hashlib
import json
from asyncio import Lock
from collections import defaultdict
from src.config.config import settings
from src.core.context import Context
from src.excetion.exceptions import InvalidParamException
from src.schema.message import ToolCall, ToolResult
from src.tools.registry import ToolHandler


class Reminder:
    _fail_cnt: defaultdict[str, defaultdict[str, int]]
    _lock: Lock

    def __init__(self):
        self._fail_cnt = defaultdict(lambda: defaultdict(int))
        self._lock = Lock()

    async def use(
            self,
            context: Context,
            tool_call: ToolCall,
            handler: ToolHandler,
    ) -> ToolResult:
        fingerprint = self.generate_fingerprint(tool_call)
        tool_result = await handler(context, tool_call)
        async with self._lock:
            if tool_result.is_error:
                cnt = self._fail_cnt[context.session_id][fingerprint] + 1
                self._fail_cnt[context.session_id][fingerprint] = cnt
                if cnt >= settings.reminder_max_count:
                    return ToolResult(
                        tool_call_id=tool_result.tool_call_id,
                        output=f"""[SYSTEM REMINDER 警告]
你似乎陷入了死循环。你刚刚连续 {cnt} 次使用相同的参数调用了 {tool_call.name} 工具，并且都失败了。
请立即停止这种无效的重试！你的注意力被当前的报错过度吸引了。
你需要：
1. 停止猜测参数。跳出当前的局部思维。
2. 彻底改变你的策略。
3. 如果你确实无法通过系统工具解决当前问题，请直接结束任务并向用户说明你需要什么人工帮助，而不是继续盲目消耗 API 资源尝试。""",
                        is_error=False,
                    )
            else:
                cnt = self._fail_cnt.get(context.session_id)
                if cnt is not None:
                    cnt.pop(fingerprint, None)
                    if not cnt:
                        del self._fail_cnt[context.session_id]
        return tool_result

    def generate_fingerprint(self, tool_call: ToolCall) -> str:
        arguments = tool_call.arguments
        if arguments is None:
            key = f"{tool_call.name}"
        elif isinstance(arguments, str):
            key = f"{tool_call.name}:{arguments}"
        elif isinstance(arguments, dict):
            key = f"{tool_call.name}:{json.dumps(arguments, ensure_ascii=False)}"
        else:
            raise InvalidParamException("Invalid tool call arguments")
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
