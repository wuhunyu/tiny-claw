import logging
import pathlib
from typing import Any

import aiofiles.os
from pydantic import BaseModel, Field, ValidationError

from src.core.context import Context
from src.excetion.exceptions import InvalidParamException, FileNotExistException, FilePermissionException
from src.schema.message import ToolDefinition
from src.util.path_util import absolute_path

logger = logging.getLogger(__name__)


class WriteFileParams(BaseModel):
    file_name: str = Field(..., description="文件名称(包括相对路径或绝对路径)")
    content: str = Field(..., description="文件内容")


class WriteFile(BaseModel):
    work_dir: str = Field(..., description="工作目录")

    def name(self) -> str:
        return "write_file"

    def readonly(self) -> bool:
        return False

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description="创建或覆盖写入一个文件。如果目录不存在会自动创建",
            readonly=self.readonly(),
            input_schema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "文件名称(包括相对路径或绝对路径)"
                    },
                    "content": {
                        "type": "string",
                        "description": "文件内容"
                    }
                },
                "required": ["file_name", "content"]
            }
        )

    async def execute(self, context: Context, arguments: dict[str, Any] | str | None) -> str:
        if not arguments:
            logger.info("请指定 file_name 和 content")
            raise InvalidParamException(message="请指定 file_name 和 content")
        if isinstance(arguments, dict):
            try:
                write_file_params = WriteFileParams.model_validate(arguments)
            except ValidationError as e:
                logger.info(f"{self.name()} 输入参数格式错误: {e}")
                raise InvalidParamException(message=f"{self.name()} 参数格式错误") from e
        elif isinstance(arguments, str):
            try:
                write_file_params = WriteFileParams.model_validate_json(arguments)
            except ValidationError as e:
                logger.info(f"{self.name()} 输入参数格式错误: {e}")
                raise InvalidParamException(message=f"{self.name()} 参数格式错误") from e
        else:
            logger.info(f"{self.name()} 格式错误")
            raise InvalidParamException(message=f"{self.name()} 格式错误")

        # 路径判断
        try:
            absolute_path_file = absolute_path(
                self.work_dir,
                write_file_params.file_name
            )
        except IOError as e:
            logger.info(f"文件不存在: {write_file_params.file_name}", exc_info=True)
            raise FileNotExistException(message=f"文件不存在: {write_file_params.file_name}") from e

        target_path = pathlib.Path(absolute_path_file)
        # 创建父目录
        try:
            await aiofiles.os.makedirs(target_path.parent, exist_ok=True)
        except PermissionError as e:
            logger.warning(f"无权限创建目录: {target_path.parent}", exc_info=True)
            raise FilePermissionException(message=f"无权限创建目录: {target_path.parent}") from e

        # 写入文件
        try:
            async with aiofiles.open(target_path, "w", encoding="utf-8") as f:
                await f.write(write_file_params.content)
        except PermissionError as e:
            logger.warning(f"无写入权限: {target_path}", exc_info=True)
            raise FilePermissionException(message=f"无写入权限: {target_path}") from e

        return "ok"
