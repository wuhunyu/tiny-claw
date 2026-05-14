import asyncio
import logging
import os
import pathlib

from src.context.composer import PromptComposer
from src.engine.loop import AgentEngine
from src.provider.chat import MyChat, Provider
from src.tools.bash import Bash
from src.tools.edit_file import EditFile
from src.tools.read_file import ReadFile
from src.tools.registry import RegistryImpl
from src.tools.write_file import WriteFile


def setup_logger():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("src").setLevel(logging.INFO)


async def main():
    # 工作区域
    work_dir = os.getenv("WORK_DIR", os.getcwd())
    # 获取绝对路径
    work_dir = str(pathlib.Path(work_dir).expanduser().resolve())

    # 初始化日志
    setup_logger()

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
        prompt_composer=PromptComposer(work_dir=work_dir),
        work_dir=work_dir,
        enable_thinking=os.getenv("ENABLE_THINKING", False),
    )

    prompt = """
我需要在当前目录下新建一个 ping.go，提供一个简单的 http ping 接口。
写完之后，帮我把代码用 git 提交一下。
    """
    await engine.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
