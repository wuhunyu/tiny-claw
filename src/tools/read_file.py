import logging
import pathlib
from typing import Any

import aiofiles
from pydantic import BaseModel, Field

from src.config.config import settings
from src.core.context import Context
from src.excetion.exceptions import InvalidParamException, FileNotExistException, NotFileException, ToolInvokeException
from src.schema.message import ToolDefinition
from src.util.path_util import absolute_path

logger = logging.getLogger(__name__)

MAX_LEN = 8000


class ReadFileParams(BaseModel):
    file_name: str = Field(..., description="文件名称")


class ReadFile(BaseModel):
    work_dir: str = Field(default=settings.work_dir, description="工作区")

    @staticmethod
    def name() -> str:
        return "read_file"

    @staticmethod
    def readonly() -> bool:
        return True

    @staticmethod
    def definition() -> ToolDefinition:
        return ToolDefinition(
            name=ReadFile.name(),
            description="读取指定文件内容",
            readonly=ReadFile.readonly(),
            input_schema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "文件名称"
                    }
                },
                "required": ["file_name"]
            }
        )

    async def execute(self, context: Context, arguments: dict[str, Any] | str | None) -> str:
        if not arguments:
            logger.info("请指定 file_name")
            raise InvalidParamException(message="请指定 file_name")
        if isinstance(arguments, dict):
            read_file_params = ReadFileParams.model_validate(arguments)
        elif isinstance(arguments, str):
            read_file_params = ReadFileParams.model_validate_json(arguments)
        else:
            logger.info(f"{self.name()} 格式错误")
            raise InvalidParamException(message=f"{self.name()} 格式错误")

        # 路径判断
        try:
            absolute_path_file = absolute_path(
                self.work_dir,
                read_file_params.file_name
            )
        except IOError as e:
            logger.info(f"文件不存在: {read_file_params.file_name}", exc_info=True)
            raise FileNotExistException(message=f"文件不存在: {read_file_params.file_name}") from e

        target_path = pathlib.Path(absolute_path_file)
        # 是否存在
        if not target_path.exists():
            logger.info(f"文件不存在: {read_file_params.file_name}")
            raise FileNotExistException(message=f"文件不存在: {read_file_params.file_name}")
        if not target_path.is_file():
            logger.info(f"不是文件: {read_file_params.file_name}")
            raise NotFileException(message=f"不是文件: {read_file_params.file_name}")
        # 执行读取任务
        try:
            async with aiofiles.open(target_path, "r") as f:
                content = await f.read()
        except Exception as e:
            logger.warning(f"读取文件内容失败: {read_file_params.file_name}", exc_info=True)
            raise ToolInvokeException(message=f"读取文件内容失败: {read_file_params.file_name}") from e

        # 超长截取
        if len(content) > MAX_LEN:
            return f"{content[:MAX_LEN]}...[由于内容过长，已被系统截断至前 {MAX_LEN} 字节]..."
        return content
