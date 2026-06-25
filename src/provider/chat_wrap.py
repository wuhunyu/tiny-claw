import logging
import time

from src.core.context import Context
from src.engine.session import Session
from src.provider.interface import LLMProvider
from src.schema.message import Message, ToolDefinition, TokenUsage

logger = logging.getLogger(__name__)

# 每百万 token 价格, 单位是元
PRICE_MODELS = {
    "gpt-5.5": {
        "input_tokens": 33.80,
        "cached_input_tokens": 3.38,
        "output_tokens": 202.80,
    },
    "gpt-5.5-cyber": {
        "input_tokens": 135.20,
        "cached_input_tokens": 13.52,
        "output_tokens": 811.20,
    },
    "gpt-5.4": {
        "input_tokens": 16.90,
        "cached_input_tokens": 1.69,
        "output_tokens": 101.40,
    },
    "gpt-5.4-mini": {
        "input_tokens": 5.07,
        "cached_input_tokens": 0.507,
        "output_tokens": 30.5552,
    },
    "gpt-5.3-codex": {
        "input_tokens": 11.83,
        "cached_input_tokens": 1.183,
        "output_tokens": 94.64,
    },
    "gpt-5.2": {
        "input_tokens": 11.83,
        "cached_input_tokens": 1.183,
        "output_tokens": 94.64,
    },
    "gpt-image-2.0-image": {
        "input_tokens": 54.08,
        "cached_input_tokens": 13.52,
        "output_tokens": 202.80,
    },
    "gpt-image-2.0-text": {
        "input_tokens": 33.80,
        "cached_input_tokens": 8.45,
        "output_tokens": 67.60,
    },
    "deepseek-v4-flash": {
        "input_tokens": 1.00,
        "cached_input_tokens": 0.02,
        "output_tokens": 2.00,
    },
    "deepseek-v4-pro": {
        "input_tokens": 3.00,
        "cached_input_tokens": 0.025,
        "output_tokens": 6.00,
    },
    "minimax-m3": {
        "input_tokens": 2.10,
        "cached_input_tokens": 0.42,
        "output_tokens": 8.40,
    },
    "minimax-m2.7": {
        "input_tokens": 2.10,
        "cached_input_tokens": 0.42,
        "output_tokens": 8.40,
    },
}

# 百万
MILLION = 1_000_000


class LLMProviderCostTrackerWrap:
    def __init__(self, provider: LLMProvider, session: Session):
        self.provider = provider
        self.session = session

    async def generate(
            self,
            context: Context,
            messages: list[Message],
            available_tools: list[ToolDefinition],
    ) -> tuple[
        Message,
        TokenUsage,
    ]:
        try:
            start = time.perf_counter()
            message, token_usage = await self.provider.generate(context, messages, available_tools)
            if token_usage and token_usage.model_name in PRICE_MODELS:
                price_model = PRICE_MODELS[token_usage.model_name]
                cached_input_tokens = price_model.get("cached_input_tokens", 0)
                input_tokens = price_model.get("input_tokens", 0)
                output_tokens = price_model.get("output_tokens", 0)
                cur_total = (
                                    token_usage.prompt_tokens * input_tokens +
                                    token_usage.completion_tokens * output_tokens
                            ) / MILLION

                # 记录 token / 费用 消耗
                await self.session.record_usage(
                    cached_prompt_tokens=cached_input_tokens,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    cost_cny=cur_total,
                )

                end = time.perf_counter()
                logger.info(
                    "[Tracker] 📊 API 调用完成 | 模型: %s | 耗时: %.2f ms | 输出缓存: %d tk | 输入: %d tk | 输出: %d tk | 花费: ¥%.6f\n",
                    token_usage.model_name,
                    (end - start) * 1000,
                    cached_input_tokens,
                    input_tokens,
                    output_tokens,
                    cur_total,
                )
            return message, token_usage
        except:
            raise
