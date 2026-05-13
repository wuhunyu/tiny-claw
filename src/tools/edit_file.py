import pathlib
from typing import Any

import aiofiles
from pydantic import BaseModel, Field, ValidationError

from src.schema.message import ToolDefinition
from src.util.path_util import absolute_path


class EditFileParams(BaseModel):
    file_name: str = Field(..., description="要修改的文件")
    old_text: str = Field(...,
                          description="文件中原有的文本。必须包含足够的上下文（建议上下各多包含几行），以确保在文件中的唯一性。")
    new_text: str = Field(..., description="要替换成的新文本")


class EditFile(BaseModel):
    work_dir: str = Field(..., description="工作目录")

    def name(self) -> str:
        return "edit_file"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description="对现有文件进行局部的字符串替换。这比重写整个文件更安全、更快速。请提供足够的 old_text 上下文以确保匹配的唯一性。",
            input_schema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "要修改的文件"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "文件中原有的文本。必须包含足够的上下文（建议上下各多包含几行），以确保在文件中的唯一性。"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "要替换成的新文本"
                    }
                },
                "required": ["file_name", "old_text", "new_text"]
            }
        )

    async def execute(self, arguments: dict[str, Any] | str | None) -> str:
        if not arguments:
            raise ValueError("请指定 file_name && old_text && new_text")
        if isinstance(arguments, dict):
            try:
                edit_file_params = EditFileParams.model_validate(arguments)
            except ValidationError as e:
                raise ValueError(f"{self.name()} 参数格式错误: {e}")
        elif isinstance(arguments, str):
            try:
                edit_file_params = EditFileParams.model_validate_json(arguments)
            except ValidationError as e:
                raise ValueError(f"{self.name()} 参数格式错误: {e}")
        else:
            raise ValueError(f"{self.name()} 格式错误")

        print(f"-> 运行 {self.name()} 命令: {arguments}")

        # 路径判断
        try:
            absolute_path_file = absolute_path(
                self.work_dir,
                edit_file_params.file_name
            )
        except IOError:
            raise

        target_path = pathlib.Path(absolute_path_file)
        # 判断文件是否存在
        if not target_path.exists():
            raise ValueError(f"文件 {edit_file_params.file_name} 不存在")
        if not target_path.is_file():
            raise ValueError(f"文件 {edit_file_params.file_name} 不是一个文件")

        async with aiofiles.open(target_path, "r+", encoding="utf-8") as f:
            original_text = await f.read()
            target_text = self._fuzzy_replace(
                original_text,
                edit_file_params.old_text,
                edit_file_params.new_text
            )
            await f.seek(0)
            await f.write(target_text)
            await f.truncate()
        return "ok"

    def _fuzzy_replace(self, original_text: str, old_text: str, new_text: str) -> str:
        # 第一阶段: 精准匹配
        count = original_text.count(old_text)
        if count == 1:
            return original_text.replace(old_text, new_text, 1)
        if count > 1:
            raise ValueError(f"old_text 匹配到了 {count} 处，请提供更多的上下文代码以确保唯一性")

        # 第二阶段: 换行符归一化(统一将 \r\n 转换成 \n)
        normalized_original_text = original_text.replace("\r\n", "\n")
        normalized_old_text = old_text.replace("\r\n", "\n")
        normalized_count = normalized_original_text.count(normalized_old_text)
        if normalized_count == 1:
            return normalized_original_text.replace(normalized_old_text, new_text, 1)

        # 第三阶段: trim space 匹配(忽略首尾的空行和空格)
        strip_normalized_old_text = normalized_old_text.strip()
        if strip_normalized_old_text:
            strip_normalized_count = normalized_original_text.count(strip_normalized_old_text)
            if strip_normalized_count == 1:
                return normalized_original_text.replace(strip_normalized_old_text, new_text, 1)

        # 第四阶段: 逐行去缩进匹配(最强力的容错: 消除大模型遗漏缩进的幻觉)
        return self._line_by_line_replace(normalized_original_text, normalized_old_text, new_text)

    def _line_by_line_replace(self, original_text: str, old_text: str, new_text: str) -> str:
        # 按行切割
        # 原文保留前后空格
        original_text_lines = original_text.split("\n")
        # 待替换文本去除前后空格
        old_text_lines = [line.strip() for line in old_text.split("\n")]
        m, n = len(original_text_lines), len(old_text_lines)
        if n == 0 or m < n:
            raise ValueError("找不到该代码片段")

        match_count = 0
        match_start = -1
        match_end = -1

        # 匹配
        for i in range(m - n + 1):
            is_match = True
            for j in range(n):
                if original_text_lines[i + j].strip() != old_text_lines[j]:
                    is_match = False
                    break
            if is_match:
                match_count += 1
                match_start = i
                match_end = i + n

        if not match_count:
            raise ValueError("在文件中未找到 old_text，请大模型先调用 read_file 仔细确认文件内容和缩进")
        if match_count > 1:
            raise ValueError(f"模糊匹配到了 {match_count} 处相似代码，请提供更多上下行代码以精确定位")

        # 拼接结果
        return "\n".join((original_text_lines[:match_start] + old_text_lines + original_text_lines[match_end:]))
