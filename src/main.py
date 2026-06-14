import asyncio
import logging

from dotenv import load_dotenv

from src.config.config import settings
from src.dingtalk.bot import create_ding_talk_bot


def setup_logger():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("src").setLevel(logging.INFO)


async def main():
    # 加载环境变量
    load_dotenv()

    # 导入配置
    _ = settings

    # 初始化日志
    setup_logger()

    # 引入 钉钉
    ding_talk_bot = await create_ding_talk_bot()
    # 启动
    await ding_talk_bot.start()


if __name__ == "__main__":
    asyncio.run(main())
