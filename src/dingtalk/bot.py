import asyncio
import logging
import uuid
from collections import defaultdict

import dingtalk_stream
from dingtalk_stream import CallbackMessage, AckMessage, DingTalkStreamClient, Credential

from src.config.config import settings
from src.context.compactor import Compactor
from src.context.composer import PromptComposer
from src.context.recovery import ToolRecoveryManager
from src.engine.loop import AgentEngine
from src.engine.session import SessionManager
from src.provider.chat import MyChat
from src.provider.interface import LLMProvider
from src.schema.message import Message, ToolCall, ToolResult
from src.tools.bash import Bash
from src.tools.edit_file import EditFile
from src.tools.read_file import ReadFile
from src.tools.registry import RegistryImpl, Registry
from src.tools.web_search_by_tavily import WebSearchByTavily
from src.tools.write_file import WriteFile

logger = logging.getLogger(__name__)


class DingTalkBotHandler(dingtalk_stream.ChatbotHandler):
    work_dir: str
    chat_client: LLMProvider
    registry: Registry
    tool_recovery_manager: ToolRecoveryManager
    prompt_composer: PromptComposer
    enable_thinking: bool
    session_manager: SessionManager
    _locks: dict[str, asyncio.Lock]

    def __init__(
            self,
            work_dir: str,
            chat_client: LLMProvider,
            registry: Registry,
            tool_recovery_manager: ToolRecoveryManager,
            prompt_composer: PromptComposer,
            enable_thinking: bool,
            session_manager: SessionManager,
    ):
        super().__init__()
        self.work_dir = work_dir
        self.chat_client = chat_client
        self.registry = registry
        self.tool_recovery_manager = tool_recovery_manager
        self.prompt_composer = prompt_composer
        self.enable_thinking = enable_thinking
        self.session_manager = session_manager
        self._locks = defaultdict(asyncio.Lock)

    async def process(self, callback: CallbackMessage):
        incoming_message = dingtalk_stream.ChatbotMessage.from_dict(callback.data)

        """
        # 群聊/私聊 类型: '1' 单聊，'2' 群聊
        conversation_type = incoming_message.conversation_type
        # 群聊/私聊 标题
        conversation_title = incoming_message.conversation_title
        # 消息类型: text / picture / richText
        message_type = incoming_message.message_type
        # 发送人id
        sender_id = incoming_message.sender_staff_id
        # 发送人昵称
        sender_nick = incoming_message.sender_nick
        """

        # 群聊/私聊 id
        conversation_id = incoming_message.conversation_id
        # 消息内容
        content_strip = incoming_message.text.content.strip()

        # 发起一个 loop
        # 实例化一个 engine
        agent_engine = AgentEngine(
            provider=self.chat_client,
            compactor=Compactor(),
            registry=self.registry,
            reporter=DingTalkBotReporter(
                handler=self,
                incoming_message=incoming_message,
            ),
            tool_recovery_manager=ToolRecoveryManager(tool_registry=self.registry),
            work_dir=self.work_dir,
            enable_thinking=self.enable_thinking,
        )
        try:
            plan_model = self.prompt_composer.plan_model or False
            logger.info(f"计划模式 (Plan Mode): {plan_model}")

            # 加载系统提示词
            system_prompt = await self.prompt_composer.build()
            # 会话id
            session_id = conversation_id or str(uuid.uuid4)
            # 获取 会话锁
            async with self._locks[session_id]:
                # 获取 session 对象
                session = await self.session_manager.get_or_create(
                    session_id=session_id,
                    work_dir=self.work_dir,
                )

                # 运行 loop
                await agent_engine.run(
                    user_prompt=content_strip,
                    system_prompt=system_prompt,
                    session=session,
                )
        except Exception:
            logger.exception("loop 运行失败")
            return AckMessage.STATUS_SYSTEM_EXCEPTION, 'error'
        return AckMessage.STATUS_OK, 'ok'


class DingTalkBotReporter:
    def __init__(
            self,
            handler: DingTalkBotHandler,
            incoming_message: dingtalk_stream.ChatbotMessage,
    ):
        self.handler = handler
        self.incoming_message = incoming_message

    def reply(self, text: str) -> None:
        self.incoming_message.at_users = []
        self.handler.reply_text(text, self.incoming_message)

    def session_start(self) -> None:
        pass

    def session_end(self) -> None:
        pass

    def step_start(self, step_count: int) -> None:
        pass

    def step_end(self, step_count: int) -> None:
        pass

    def on_thinking(self, message: Message) -> None:
        self.reply(f"思考结果: {message.content}")

    def pre_tool_call(self, tool_call: ToolCall) -> None:
        self.reply(f"🛠️ 执行工具: {tool_call.name}, 参数: {tool_call.arguments}")

    def post_tool_call(self, tool_result: ToolResult) -> None:
        pass

    def on_message(self, message: Message) -> None:
        self.reply(f"{message.content}")


class DingTalkBot:
    dingtalk_stream_client: DingTalkStreamClient
    work_dir: str

    def __init__(
            self,
            dingtalk_stream_client: DingTalkStreamClient,
            work_dir: str,
    ):
        super().__init__()
        self.dingtalk_stream_client = dingtalk_stream_client
        self.work_dir = work_dir

    async def start(self):
        await self.dingtalk_stream_client.start()


async def create_ding_talk_bot() -> DingTalkBot:
    # 工作区域
    work_dir = settings.work_dir

    # llm client
    chat_client = MyChat(
        llm_provider=settings.llm_provider,
    )

    # 工具注册中心
    registry = RegistryImpl()
    await registry.registry(tool=ReadFile(work_dir=work_dir))
    await registry.registry(tool=WriteFile(work_dir=work_dir))
    await registry.registry(
        tool=Bash(
            work_dir=work_dir,
            timeout=settings.bash_timeout,
        ),
    )
    await registry.registry(tool=EditFile(work_dir=work_dir))
    await registry.registry(tool=WebSearchByTavily())

    # 工具恢复管理器
    tool_recovery_manager = ToolRecoveryManager(tool_registry=registry)

    # 提示词组合器
    prompt_composer = PromptComposer(work_dir=work_dir, plan_model=False)

    # 钉钉客户端
    dingtalk_stream_client = DingTalkStreamClient(
        Credential(
            settings.dingtalk_client_id,
            settings.dingtalk_client_secret,
        ),
    )
    dingtalk_stream_client.register_callback_handler(
        dingtalk_stream.chatbot.ChatbotMessage.TOPIC,
        DingTalkBotHandler(
            work_dir=work_dir,
            chat_client=chat_client,
            registry=registry,
            tool_recovery_manager=tool_recovery_manager,
            prompt_composer=prompt_composer,
            enable_thinking=settings.enable_thinking or False,
            session_manager=SessionManager(),
        )
    )

    return DingTalkBot(
        dingtalk_stream_client=dingtalk_stream_client,
        work_dir=work_dir,
    )
