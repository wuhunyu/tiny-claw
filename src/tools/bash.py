import asyncio
import os
import signal
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.schema.message import ToolDefinition

MAX_LEN = 8000


class BashParams(BaseModel):
    command: list[str] = Field(..., description="bash 命令")


class Bash(BaseModel):
    work_dir: str = Field(..., description="工作目录")
    timeout: int = Field(default=30, description="超时时间(秒)")

    def name(self) -> str:
        return "bash"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description="执行 bash 命令",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "bash 结构化命令"
                        }
                    }
                },
                "required": ["command"]
            }
        )

    async def execute(self, arguments: dict[str, Any] | str | None) -> str:
        if not arguments:
            raise ValueError("请指定 command")
        if isinstance(arguments, dict):
            try:
                bash_params = BashParams.model_validate(arguments)
            except ValidationError as e:
                raise ValueError(f"{self.name()} 参数格式错误: {e}")
        elif isinstance(arguments, str):
            try:
                bash_params = BashParams.model_validate_json(arguments)
            except ValidationError as e:
                raise ValueError(f"{self.name()} 参数格式错误: {e}")
        else:
            raise ValueError("command 格式错误")

        print(f"-> 运行 bash 命令: {' '.join(bash_params.command)}")

        # 创建子进程执行
        try:
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", *bash_params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # 指定运行目录为工作空间
                cwd=self.work_dir,
                preexec_fn=os.setsid,
            )
        except Exception as e:
            raise ValueError(f"创建子进程失败: {bash_params.command}", e)

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                # 设置超时时间
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            self._kill_process_group(process)
            await process.wait()
            return f"执行超时({self.timeout}s): {bash_params.command}"
        except Exception as e:
            self._kill_process_group(process)
            await process.wait()
            return f"执行失败: {bash_params.command}\n{e}"

        stdout = stdout_bytes.decode('utf-8', errors='replace')
        stderr = stderr_bytes.decode('utf-8', errors='replace')
        if process.returncode != 0:
            # 错误原样返回, 依赖 Self-Correction 自愈机制
            return f"执行失败: {bash_params.command}\n{stderr}"
        if not stdout:
            return f"执行成功: {bash_params.command}"
        # 超长截断
        if len(stdout) > MAX_LEN:
            print(f"-> 执行成功: {stdout[:MAX_LEN]}")
            return f"执行成功: {stdout[:MAX_LEN]}...[由于内容过长，已被系统截断至前 {MAX_LEN} 字节]..."
        print(f"-> 执行成功: {stdout}")
        return f"执行成功: {bash_params.command}\n{stdout}"

    def _kill_process_group(self, process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
