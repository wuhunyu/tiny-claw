import pathlib
from typing import Any

import aiofiles.os
from pydantic import BaseModel, Field, ValidationError

from src.schema.message import ToolDefinition


class WriteFileParams(BaseModel):
    file_name: str = Field(..., description="文件名称(包括相对路径或绝对路径)")
    content: str = Field(..., description="文件内容")


class WriteFile(BaseModel):
    work_dir: str = Field(..., description="工作目录")

    def name(self) -> str:
        return "write_file"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description="创建或覆盖写入一个文件。如果目录不存在会自动创建",
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

    async def execute(self, arguments: dict[str, Any] | str | None) -> str:
        if not arguments:
            raise ValueError("请指定 file_name 和 content")
        if isinstance(arguments, dict):
            try:
                write_file_params = WriteFileParams.model_validate(arguments)
            except ValidationError as e:
                raise ValueError(f"{self.name()} 参数格式错误: {e}")
        elif isinstance(arguments, str):
            try:
                write_file_params = WriteFileParams.model_validate_json(arguments)
            except ValidationError as e:
                raise ValueError(f"{self.name()} 参数格式错误: {e}")
        else:
            raise ValueError("file_name 和 content 格式错误")

        # 路径判断
        absolute_work_dir = pathlib.Path(self.work_dir).expanduser().resolve()
        input_path = pathlib.Path(write_file_params.file_name).expanduser()
        if not input_path.is_absolute():
            input_path = absolute_work_dir / input_path
        target_path = input_path.resolve()
        if not target_path.is_relative_to(absolute_work_dir):
            raise ValueError(f"文件 {write_file_params.file_name} 不在工作区 {self.work_dir} 下")

        # 创建父目录
        try:
            await aiofiles.os.makedirs(target_path.parent, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"无权限创建目录: {target_path.parent}") from e

        # 写入文件
        try:
            async with aiofiles.open(target_path, "w", encoding="utf-8") as f:
                await f.write(write_file_params.content)
        except PermissionError as e:
            raise PermissionError(f"无写入权限: {target_path}") from e

        return "ok"
