import json
import logging
from enum import Enum
from typing import Any

from litellm.responses.main import aresponses
from pydantic import BaseModel, Field, ValidationError

from src.config.config import settings
from src.excetion.exceptions import TinyClawException, InvalidParamException, ResponseException, ResponseBlankException
from src.provider.interface import Provider
from src.schema.message import ToolCall, Message, Role
from src.tools.registry import Registry

logger = logging.getLogger(__name__)


class ToolRecoveryAction(str, Enum):
    RETRY = "retry"
    ASK_USER = "ask_user"
    SKIP = "skip"
    FAIL = "fail"


class ToolRecoveryConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolRecovery(BaseModel):
    action: ToolRecoveryAction = Field(..., description="工具恢复动作")
    reason: str = Field(..., description="简短说明为什么这么恢复")
    tool_call: ToolCall = Field(..., description="如果 action=retry，则这里放修正后的 ToolCall")
    question: str | None = Field(default=None, description="如果 action=ask_user，则这里放要问用户的问题")
    confidence: ToolRecoveryConfidence = Field(default=ToolRecoveryConfidence.MEDIUM, description="工具恢复的置信度")

    def to_user_message(self) -> str:
        lines = [
            self._message_title(),
            f"- 恢复动作: {self.action.value}",
            f"- 恢复原因: {self.reason}",
            f"- 置信度: {self.confidence.value}",
            *self._extra_message_lines(),
        ]

        return "\n".join(lines)

    def _message_title(self) -> str:
        title_map = {
            ToolRecoveryAction.RETRY: "工具调用恢复建议：",
            ToolRecoveryAction.ASK_USER: "工具调用恢复需要用户补充信息：",
            ToolRecoveryAction.SKIP: "工具调用恢复建议：",
            ToolRecoveryAction.FAIL: "工具调用恢复失败：",
        }
        return title_map.get(self.action, "工具调用恢复结果：")

    def _extra_message_lines(self) -> list[str]:
        if self.action == ToolRecoveryAction.RETRY:
            return [f"- 建议重新调用工具: {self.tool_call.name}"]

        if self.action == ToolRecoveryAction.ASK_USER:
            return [f"- 需要询问用户: {self.question or '未提供具体问题'}"]

        return []

    @staticmethod
    def to_response_schema() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "tool_recovery",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "工具恢复动作",
                            "enum": ["retry", "ask_user", "skip", "fail"],
                        },
                        "reason": {
                            "type": "string",
                            "description": "简短说明为什么这么恢复",
                        },
                        "tool_call": {
                            "description": "如果 action=retry，则这里放修正后的 ToolCall；否则为 null",
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "要调用的工具名称",
                                        },
                                        "arguments": {
                                            "type": "object",
                                            "description": "工具调用参数",
                                            "additionalProperties": True,
                                        },
                                    },
                                    "required": ["name", "arguments"],
                                },
                                {
                                    "type": "null",
                                },
                            ],
                        },
                        "question": {
                            "type": ["string", "null"],
                            "description": "如果 action=ask_user，则这里放要问用户的问题；否则为 null",
                        },
                        "confidence": {
                            "type": "string",
                            "description": "工具恢复的置信度",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": [
                        "action",
                        "reason",
                        "tool_call",
                        "question",
                        "confidence",
                    ],
                },
            },
        }


class ToolRecoveryLLM(BaseModel):
    llm_provider: Provider = Field(default=Provider.OPENAI, description="模型供应商")
    base_url: str = Field(default_factory=lambda: settings.llm_base_url, description="模型供应商的 API 地址")
    api_key: str = Field(default_factory=lambda: settings.llm_api_key, description="模型供应商的 API Key")
    model: str = Field(default_factory=lambda: settings.llm_model or "gpt-5.3-codex", description="模型名称")

    async def generate(
            self,
            messages,
            tools,
    ) -> ToolRecovery:
        if not messages:
            raise InvalidParamException(message="messages 不能为空")
        tools = tools or []
        try:
            resp = await aresponses(
                url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                input=messages,
                tools=tools,
                text_format=ToolRecovery.to_response_schema(),
            )
        except Exception as e:
            logger.exception(f"{self.llm_provider.value} Recovery API 响应失败")
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

    你的任务是：根据一次失败的工具调用 tool_call 和异常 cause，判断应该如何恢复。

    你只能给出恢复建议，不能执行任何工具调用，不能发起新的 tool call，不能声称自己已经调用了工具。

    你必须从以下动作中选择一个：
    - retry: 原工具调用可以通过修正参数、名称或格式后重试
    - ask_user: 缺少必要信息，必须向用户追问
    - skip: 当前工具失败不影响主流程，可以跳过
    - fail: 该错误无法安全恢复，应该终止当前流程

    决策规则：
    1. 如果异常明显是参数名错误、参数类型错误、JSON 格式错误、缺少可从上下文推断的参数，优先选择 retry。
    2. 如果缺少的信息无法从 tool_call 或 cause 中可靠推断，选择 ask_user。
    3. 如果错误说明该工具结果非必要，或者失败不影响继续完成任务，选择 skip。
    4. 如果错误涉及权限、认证、不可恢复的系统错误、危险操作、重复失败或数据不一致，选择 fail。
    5. 不要编造不存在的参数值。
    6. 不要输出多余解释，只输出符合响应 schema 的结果。
    7. 当 action=retry 时，tool_call 必须是修正后的工具调用，question 必须为 null。
    8. 当 action=ask_user 时，question 必须是要问用户的问题，tool_call 必须为 null。
    9. 当 action=skip 或 action=fail 时，tool_call 和 question 都必须为 null。
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
            tool_call: ToolCall,
            error_description: str,
    ) -> str | None:
        if not error_description:
            return None
        # 消息内容
        messages = [
            ("system", ToolRecoveryManager.TOOL_RECOVERY_SYSTEM_PROMPT),
            ("user", self._build_tool_recovery_user_prompt(tool_call=tool_call, error_description=error_description)),
        ]

        # 可用工具
        tool_defines = []
        for available_tool in await self.tool_registry.get_available_tools():
            tool_defines.append({
                "type": "function",
                "function": {
                    "name": available_tool.name,
                    "description": available_tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": available_tool.input_schema.get("properties", {}),
                        "required": available_tool.input_schema.get("required", []),
                        "additionalProperties": False
                    },
                    "strict": True
                }
            })

        tool_recovery = await self.tool_recovery_llm.generate(
            messages=messages,
            tools=tool_defines,
        )
        return tool_recovery.to_user_message()

    def _build_tool_recovery_user_prompt(
            self,
            tool_call: ToolCall,
            error_description: str,
    ) -> str:
        payload: dict[str, Any] = {
            "failed_tool_call": {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            "error_description": error_description,
            "instruction": (
                "请根据 failed_tool_call 和 error_description 判断恢复策略。"
                "你只能给出建议，不能调用工具，不能模拟工具执行结果。"
            ),
        }

        return json.dumps(payload, ensure_ascii=False)
