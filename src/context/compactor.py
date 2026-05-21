import logging

import tiktoken
from tiktoken import Encoding

from src.config.config import settings
from src.schema.message import Message, Role

logger = logging.getLogger(__name__)


class Compactor:
    model_context_window_tokens: int
    compact_watermark_ratio: float
    working_memory_msgs: int
    encoding: Encoding

    def __init__(
            self,
            model_context_window_tokens: int | None = None,
            compact_watermark_ratio: float | None = None,
            working_memory_msgs: int | None = None,
            model_name: str | None = None,
    ):
        self.model_context_window_tokens = model_context_window_tokens or settings.model_context_window_tokens
        self.compact_watermark_ratio = compact_watermark_ratio or settings.compact_watermark_ratio
        self.working_memory_msgs = working_memory_msgs or settings.working_memory_msgs
        model_name = model_name or settings.llm_model
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # fallback to o200k_base
            self.encoding = tiktoken.get_encoding("o200k_base")

    def compact(self, messages: list[Message]) -> list[Message]:
        # 估算消息的 token 长度
        tokens = self.estimate_message_tokens(messages)
        if tokens <= self.model_context_window_tokens * self.compact_watermark_ratio:
            return messages

        logger.info(
            f"[Compactor] ⚠️ 内存告警：当前上下文长度 ({tokens} tokens) 超过阈值 ({self.model_context_window_tokens * self.compact_watermark_ratio})，触发压缩清理...")

        message_count = len(messages)
        protect_start_index = message_count - self.working_memory_msgs
        protect_start_index = 0 if protect_start_index < 0 else protect_start_index
        ans = []
        for i, message in enumerate(messages):
            message_copy = message.model_copy(deep=True)

            if message_copy.role == Role.ROLE_SYSTEM or not message_copy.content:
                # 系统消息 和 不压缩 tool call 消息
                ans.append(message_copy)
                continue
            # 当前消息是否在 working_memory 中
            is_not_in_working_memory = i < protect_start_index
            origin_content_len = len(self.encoding.encode(message_copy.content))
            # 对于 tool call result
            if message_copy.role == Role.ROLE_TOOL:
                if is_not_in_working_memory:
                    if origin_content_len > 200:
                        message_copy.content = f"...[为了节省内存，早期的工具输出已被系统强制清理。原始长度: {origin_content_len} tokens]..."
                else:
                    if origin_content_len > 1000:
                        # 简单处理一下
                        head = message_copy.content[:500]
                        mid = message_copy.content[500:-500]
                        tail = message_copy.content[-500:]
                        message_copy.content = f"{head}\n\n...[内容过长，中间 {len(self.encoding.encode(mid))} tokens 已被系统截断]...\n\n{tail}"
            elif message_copy.role == Role.ROLE_ASSISTANT and message_copy.content:
                if is_not_in_working_memory and origin_content_len > 200:
                    message_copy.content = "...[早期的推理思考过程已折叠]..."
                    if message_copy.reasoning_content:
                        message_copy.reasoning_content = "...[早期的推理思考过程已折叠]..."
            ans.append(message_copy)

        # 估算压缩之后的消息的 token 长度
        compact_tokens = self.estimate_message_tokens(ans)
        logger.info(f"[Compactor] ✅ 压缩完成。上下文长度从 {tokens} tokens 降至 {compact_tokens} tokens")
        return ans

    def estimate_message_tokens(self, messages: list[Message]) -> int:
        if not messages:
            return 0
        text = []
        for message in messages:
            if message.content:
                text.append(message.content)
            if message.reasoning_content:
                text.append(message.reasoning_content)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    text.append(
                        f"{tool_call.id}\n{tool_call.name}\n{tool_call.arguments if tool_call.arguments else ''}")
        return len(self.encoding.encode("\n".join(text)))
