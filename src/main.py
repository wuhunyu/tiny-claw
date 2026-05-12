import asyncio
import os
import pathlib

from src.engine.loop import AgentEngine
from src.provider.chat import MyChat, Provider
from src.tools.bash import Bash
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

    # agent
    engine = AgentEngine(
        provider=chat_client,
        registry=registry,
        work_dir=work_dir,
        enable_thinking=os.getenv("ENABLE_THINKING", False),
    )

    prompt = """
    请帮我执行以下操作： 
    1. 用 bash 查看一下我当前电脑的 Go 版本。 
    2. 帮我写一个简单的 helloworld.go 文件，输出 "Hello, go-tiny-claw!"。 
    3. 用 bash 编译并运行这个 go 文件，确认它能正常工作。
    """
    await engine.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
