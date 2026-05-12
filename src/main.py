import asyncio
import os

from src.engine.loop import AgentEngine
from src.provider.chat import MyChat, Provider
from src.tools.read_file import ReadFile
from src.tools.registry import RegistryImpl


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

    # agent
    engine = AgentEngine(
        provider=chat_client,
        registry=registry,
        work_dir=work_dir,
        enable_thinking=os.getenv("ENABLE_THINKING", False),
    )

    prompt = """
    我想要知道 ~/WorkSpace/Python/tiny-claw/src/main.py 这个文件的内容是什么?
    """
    await engine.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
