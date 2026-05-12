import json
import os
import pathlib
from tarfile import AbsoluteLinkError
from typing import Any

import aiofiles
from pydantic import BaseModel, Field

from src.schema.message import ToolDefinition

MAX_LEN = 8000


class ReadFile(BaseModel):
    work_dir: str = Field(default=os.getcwd(), description="工作区")

    def name(self) -> str:
        return "read_file"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description="读取指定文件内容",
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

    async def execute(self, arguments: dict[str, Any] | str | None) -> str:
        if not arguments:
            raise ValueError("请指定 file_name")
        if isinstance(arguments, dict):
            file_name = arguments.get("file_name", None)
        elif isinstance(arguments, str):
            file_name = json.loads(arguments).get("file_name", None)
        else:
            raise ValueError("file_name 格式错误")
        if not file_name:
            raise ValueError("请指定 file_name")

        absolute_work_dir = pathlib.Path(self.work_dir).expanduser().resolve()
        input_path = pathlib.Path(file_name).expanduser()
        if not input_path.is_absolute():
            input_path = absolute_work_dir / input_path
        target_path = input_path.resolve()
        if not target_path.is_relative_to(absolute_work_dir):
            raise ValueError(f"文件 {file_name} 不在工作区 {self.work_dir} 下")

        # 是否存在
        if not target_path.exists():
            raise ValueError(f"文件不存在: {file_name}")
        if not target_path.is_file():
            raise ValueError(f"不是文件: {file_name}")
        # 执行读取任务
        try:
            async with aiofiles.open(target_path, "r") as f:
                content = await f.read()
        except Exception as e:
            raise ValueError(f"读取文件内容失败: {file_name}", e)

        # 超长截取
        if len(content) > MAX_LEN:
            return f"{content[:MAX_LEN]}...[由于内容过长，已被系统截断至前 {MAX_LEN} 字节]..."
        return content
