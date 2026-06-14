import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.core.context import Context
from src.engine.channel import ChannelMessage
from src.excetion.exceptions import InvalidParamException, ToolInvokeException
from src.schema.message import ToolDefinition

logger = logging.getLogger(__name__)


class AskUserParams(BaseModel):
    question: str = Field(..., description="问题或者需要用户进行决策的方向")


class AskUser:
    def __init__(self, channel_message: ChannelMessage):
        self.channel_message = channel_message

    def name(self) -> str:
        return "ask_user"

    def readonly(self) -> bool:
        return False

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description="需要用户进行决策, 遇到无法解决的问题, 或者需要执行高危动作时, 请求用户的帮助. 问题需要凝练, 且只能在其它 tool 无法处理时调用",
            readonly=self.readonly(),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户需要进行决策的方向, 无法解决的问题, 或者需要执行高危动作审核"
                    }
                },
                "required": ["question"]
            }
        )

    async def execute(self, context: Context, arguments: dict[str, Any] | str | None) -> str:
        if not arguments:
            logger.info("请指定 question")
            raise InvalidParamException(message="请指定 question")
        if isinstance(arguments, dict):
            try:
                ask_user_params = AskUserParams.model_validate(arguments)
            except ValidationError as e:
                logger.info(f"{self.name()} 输入参数格式错误: {e}")
                raise InvalidParamException(message=f"{self.name()} 参数格式错误: {e}")
        elif isinstance(arguments, str):
            try:
                ask_user_params = AskUserParams.model_validate_json(arguments)
            except ValidationError as e:
                logger.info(f"{self.name()} 参数格式错误: {e}")
                raise InvalidParamException(message=f"{self.name()} 参数格式错误: {e}")
        else:
            logger.warning(f"{self.name()} 格式错误")
            raise InvalidParamException(message=f"{self.name()} 格式错误")

        # 发送并接收用户输入
        answer = await self.channel_message.send_and_receive(
            context,
            ask_user_params.question,
        )
        if not answer:
            raise ToolInvokeException(message=f"{self.name()} 未接收到任务回复")
        return answer
