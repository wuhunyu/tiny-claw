import asyncio
import os
import pathlib

from src.engine.loop import AgentEngine
from src.provider.chat import MyChat, Provider
from src.tools.bash import Bash
from src.tools.edit_file import EditFile
from src.tools.read_file import ReadFile
from src.tools.registry import RegistryImpl
from src.tools.write_file import WriteFile


async def main():
    # 工作区域
    work_dir = os.getenv("WORK_DIR", os.getcwd())
    # 获取绝对路径
    work_dir = str(pathlib.Path(work_dir).expanduser().resolve())
    print(f"当前工作区: {work_dir}")

    # llm client
    chat_client = MyChat(
        llm_provider=Provider.OPENAI,
    )

    # 工具注册中心
    registry = RegistryImpl()
    await registry.registry(tool=ReadFile(work_dir=work_dir))
    await registry.registry(tool=WriteFile(work_dir=work_dir))
    await registry.registry(tool=Bash(work_dir=work_dir, timeout=30))
    await registry.registry(tool=EditFile(work_dir=work_dir))

    # agent
    engine = AgentEngine(
        provider=chat_client,
        registry=registry,
        work_dir=work_dir,
        enable_thinking=os.getenv("ENABLE_THINKING", False),
    )

    prompt = """
    当前目录下有一个 message.go 文件
    请完善代码注释
    """
    await engine.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
