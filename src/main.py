import asyncio
import os

from src.engine.loop import AgentEngine
from src.provider.chat import MyChat, Provider
from src.tools.read_file import ReadFile
from src.tools.registry import RegistryImpl
from src.tools.write_file import WriteFile


async def main():
    # 工作区域
    work_dir = os.getenv("WORK_DIR", os.getcwd())
    print(f"当前工作区: {work_dir}")

    # llm client
    chat_client = MyChat(
        llm_provider=Provider.OPENAI,
    )

    # 工具注册中心
    registry = RegistryImpl()
    await registry.registry(tool=ReadFile(work_dir=work_dir))
    await registry.registry(tool=WriteFile(work_dir=work_dir))

    # agent
    engine = AgentEngine(
        provider=chat_client,
        registry=registry,
        work_dir=work_dir,
        enable_thinking=os.getenv("ENABLE_THINKING", False),
    )

    prompt = """
    请写一个 Go 版本的 Hello World 程序
    并把结果写入到 ~/Desktop/main.go 中
    """
    await engine.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
