import json
import logging
import traceback
from enum import Enum
from typing import Any

from litellm.responses.main import aresponses
from pydantic import BaseModel, Field, ValidationError

import langsmith as ls

from src.config.config import settings
from src.core.context import Context
from src.excetion.exceptions import InvalidParamException, ResponseException, ResponseBlankException
from src.provider.interface import Provider
from src.schema.message import ToolCall
from src.tools.registry import Registry

logger = logging.getLogger(__name__)


class ToolRecoveryAction(str, Enum):
    RETRY = "retry"
    SKIP = "skip"
    FAIL = "fail"


class ToolRecoveryConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolRecovery(BaseModel):
    action: ToolRecoveryAction = Field(..., description="工具恢复动作")
    reason: str = Field(..., description="简短说明为什么这么恢复")
    tool_call: ToolCall | None = Field(..., description="如果 action=retry，则这里放修正后的 ToolCall")
    confidence: ToolRecoveryConfidence = Field(default=ToolRecoveryConfidence.MEDIUM, description="工具恢复的置信度")

    def to_user_message(self, origin_error: str) -> str:
        lines = [
            f"执行 tool 遇到如下错误: {origin_error}",
            "\n",
            "请按如下方案尝试进行恢复:",
            "当 置信度 为 high 或 medium 时, 请严格按照如下恢复方案进行恢复尝试",
            "当 置信度 为 low 时, 恢复方案作为参考意见",
            f"- 恢复动作: {self.action.value}",
            f"- 恢复原因: {self.reason}",
            f"- 置信度: {self.confidence.value}",
            *self._extra_message_lines(),
        ]

        return "\n".join(lines)

    def _extra_message_lines(self) -> list[str]:
        if self.action == ToolRecoveryAction.RETRY:
            return [
                f"- 建议调用 tool, tool_name: {self.tool_call.name}, arguments: {json.dumps(self.tool_call.arguments, ensure_ascii=False)}"]

        return []

    @staticmethod
    def to_response_schema() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "$defs": {
                        "ToolCall": {
                            "properties": {
                                "id": {
                                    "description": "工具调用ID",
                                    "title": "Id",
                                    "type": "string"
                                },
                                "name": {
                                    "description": "工具名称",
                                    "title": "Name",
                                    "type": "string"
                                },
                                "arguments": {
                                    "anyOf": [
                                        {
                                            "type": "string"
                                        },
                                        {
                                            "type": "null"
                                        }
                                    ],
                                    "description": "工具调用参数",
                                    "title": "Arguments"
                                }
                            },
                            "required": [
                                "id",
                                "name",
                                "arguments"
                            ],
                            "title": "ToolCall",
                            "type": "object",
                            "additionalProperties": False
                        },
                        "ToolRecoveryAction": {
                            "enum": [
                                "retry",
                                "skip",
                                "fail"
                            ],
                            "title": "ToolRecoveryAction",
                            "type": "string"
                        },
                        "ToolRecoveryConfidence": {
                            "enum": [
                                "low",
                                "medium",
                                "high"
                            ],
                            "title": "ToolRecoveryConfidence",
                            "type": "string"
                        }
                    },
                    "properties": {
                        "action": {
                            "description": "工具恢复动作",
                            "enum": [
                                "retry",
                                "skip",
                                "fail"
                            ],
                            "title": "ToolRecoveryAction",
                            "type": "string"
                        },
                        "reason": {
                            "description": "简短说明为什么这么恢复",
                            "title": "Reason",
                            "type": "string"
                        },
                        "tool_call": {
                            "anyOf": [
                                {
                                    "$ref": "#/$defs/ToolCall"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "description": "如果 action=retry，则这里放修正后的 ToolCall"
                        },
                        "confidence": {
                            "default": "medium",
                            "description": "工具恢复的置信度",
                            "enum": [
                                "low",
                                "medium",
                                "high"
                            ],
                            "title": "ToolRecoveryConfidence",
                            "type": "string"
                        }
                    },
                    "required": [
                        "action",
                        "reason",
                        "tool_call",
                        "confidence"
                    ],
                    "title": "ToolRecovery",
                    "type": "object",
                    "additionalProperties": False
                },
                "name": "ToolRecovery",
                "strict": True
            }
        }


class ToolRecoveryLLM(BaseModel):
    llm_provider: Provider = Field(default=Provider.OPENAI, description="模型供应商")
    base_url: str = Field(default_factory=lambda: settings.llm_base_url, description="模型供应商的 API 地址")
    api_key: str = Field(default_factory=lambda: settings.llm_api_key, description="模型供应商的 API Key")
    model: str = Field(default_factory=lambda: settings.llm_model or "gpt-5.3-codex", description="模型名称")

    async def generate(
            self,
            context: Context,
            messages,
            tools,
    ) -> ToolRecovery:
        if not messages:
            raise InvalidParamException(message="messages 不能为空")
        tools = tools or []

        with ls.trace(
                run_type="llm",
                name=f"recovery {self.model}",
                inputs={
                    "message": messages,
                    "tools": tools,
                },
                metadata={
                    "tiny_claw_session_id": context.session_id,
                    "base_url": self.base_url,
                    "model": self.model,
                },
                tags=["llm", "recovery"],
        ) as run:
            try:
                resp = await aresponses(
                    api_base=self.base_url,
                    api_key=self.api_key,
                    model=self.model,
                    input=messages,
                    tools=tools,
                    text_format=ToolRecovery.to_response_schema(),
                )
                run.end(outputs={
                    "llm_res": resp,
                })
            except Exception as e:
                logger.exception(f"{self.llm_provider.value} Recovery API 响应失败")
                run.end(error=traceback.format_exc())
                raise ResponseException(message=f"{self.llm_provider.value} Recovery API 响应失败") from e

        if not resp.output_text:
            logger.warning(f"{self.llm_provider.value} Recovery API 响应结果为空")
            raise ResponseBlankException(message=f"{self.llm_provider.value} Recovery API 响应结果为空")

        try:
            return ToolRecovery.model_validate_json(resp.output_text)
        except ValidationError as e:
            logger.exception(f"提取工具恢复结果失败")
            raise ResponseException(message=f"提取工具恢复结果失败") from e


class ToolRecoveryManager:
    TOOL_RECOVERY_SYSTEM_PROMPT = """
你是一个工具调用异常恢复顾问。

你的任务是：根据一次失败的工具调用 failed_tool_call 和异常 error_description，判断应该如何恢复。

你只能给出恢复建议，不能执行任何工具调用，不能发起新的 tool call，不能声称自己已经调用了工具。

你必须从以下动作中选择一个：

* retry: 可以通过修正工具名称、参数、参数格式，或改为调用 ask_user 后继续流程
* skip: 当前工具失败完全不影响主流程，可以安全跳过
* fail: 该错误无法安全恢复，应该终止当前流程

重要约束：

1. ask_user 不是 action。不要输出 action=ask_user。
2. 如果需要向用户提问，必须输出 action=retry，并将 tool_call 设置为 name=ask_user 的工具调用。
3. 当 action=retry 时，tool_call 必须是下一步建议调用的工具。
4. 当 action=skip 或 action=fail 时，tool_call 必须为 null。
5. 不要编造不存在的参数值。
6. 不要输出多余解释，只输出符合响应 schema 的结果。

恢复优先级，必须按顺序判断：

1. 如果原工具调用可以通过修正参数名、参数类型、JSON 格式、工具名称、或补全可从上下文可靠推断的参数来恢复，选择 retry，并返回修正后的原工具调用。
2. 如果缺少必要信息，且该信息无法从 failed_tool_call 或 error_description 中可靠推断，但用户可以提供该信息，选择 retry，并返回 ask_user 工具调用。
3. 如果需要用户确认、授权、选择方案，或需要用户审核高危动作，选择 retry，并返回 ask_user 工具调用。
4. 只有当工具失败对主流程没有实质影响，且跳过后仍能继续完成用户目标时，才选择 skip。
5. 只有当错误无法通过修正参数、询问用户、跳过非必要步骤来安全恢复时，才选择 fail。

fail 只能用于以下情况：

* 权限或认证错误，且无法通过询问用户在当前流程中解决
* 工具或系统发生不可恢复的内部错误
* 继续执行会造成明确的数据不一致、重复危险操作或不可接受风险
* 用户已经拒绝必要授权或必要确认
* 同一错误已经重复恢复失败
* 当前任务本身无法完成，且询问用户也无法提供有效解决路径

skip 只能用于以下情况：

* 当前工具结果是可选信息
* 当前工具失败不影响继续完成主任务
* 当前工具只是辅助增强体验，缺失后不会导致错误结论或错误操作

ask_user 的使用规则：

* 尽量不要调用 ask_user；只有在自动恢复不可靠时才调用。
* 但当缺少的信息、确认、授权或选择必须由用户提供时，必须调用 ask_user，而不是 fail。
* ask_user 的 question 必须简洁、明确、可回答。
* question 不要包含冗长背景，不要一次询问多个无关问题。
* 如果需要用户在多个选项中选择，应清楚列出选项。

决策示例：

* 参数 JSON 格式错误，可以修复：action=retry，tool_call=修复后的原工具调用
* 缺少 user_id，但上下文里能确定 user_id：action=retry，tool_call=补全 user_id 的原工具调用
* 缺少 user_id，且上下文无法确定，但用户可以提供：action=retry，tool_call=ask_user
* 删除、覆盖、付款、发送外部消息等高危动作需要确认：action=retry，tool_call=ask_user
* 查询天气失败，但主任务是写一封邮件，天气只是可选补充：action=skip，tool_call=null
* API key 无效，当前流程无法修复，询问用户也不能在工具调用内解决：action=fail，tool_call=null
* 工具内部服务崩溃或返回不可恢复系统错误：action=fail，tool_call=null

"""

    tool_registry: Registry
    tool_recovery_llm: ToolRecoveryLLM

    def __init__(self, tool_registry: Registry):
        self.tool_registry = tool_registry
        self.tool_recovery_llm = ToolRecoveryLLM(
            llm_provider=settings.llm_provider,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )

    async def analyze(
            self,
            context: Context,
            tool_call: ToolCall,
            error_description: str,
    ) -> str | None:
        if not error_description:
            return None
        # 消息内容
        messages = [
            {
                "role": "system",
                "content": ToolRecoveryManager.TOOL_RECOVERY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self._build_tool_recovery_user_prompt(
                    tool_call=tool_call,
                    error_description=error_description
                ),
            },
        ]

        # 可用工具
        tool_defines = []
        for available_tool in await self.tool_registry.get_available_tools():
            tool_defines.append({
                "type": "function",
                "name": available_tool.name,
                "description": available_tool.description,
                "parameters": {
                    "type": "object",
                    "properties": available_tool.input_schema.get("properties", {}),
                    "required": available_tool.input_schema.get("required", []),
                    "additionalProperties": False
                },
                "strict": True
            })

        tool_recovery = await self.tool_recovery_llm.generate(
            context=context,
            messages=messages,
            tools=tool_defines,
        )
        return tool_recovery.to_user_message(error_description)

    def _build_tool_recovery_user_prompt(
            self,
            tool_call: ToolCall,
            error_description: str,
    ) -> str:
        payload: dict[str, Any] = {
            "tool_call": {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            "error_description": error_description,
            "instruction": (
                "请根据 tool_call 和 error_description 判断恢复策略。"
                "你只能给出建议，不能调用工具，不能模拟工具执行结果。"
            ),
        }

        return json.dumps(payload, ensure_ascii=False)
