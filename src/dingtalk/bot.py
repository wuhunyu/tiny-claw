import logging
import os
import pathlib

import dingtalk_stream
from dingtalk_stream import CallbackMessage, AckMessage, DingTalkStreamClient, Credential

from src.context.composer import PromptComposer
from src.engine.loop import AgentEngine
from src.provider.chat import MyChat, Provider
from src.provider.interface import LLMProvider
from src.schema.message import Message, ToolCall, ToolResult
from src.tools.bash import Bash
from src.tools.edit_file import EditFile
from src.tools.read_file import ReadFile
from src.tools.registry import RegistryImpl, Registry
from src.tools.write_file import WriteFile

logger = logging.getLogger(__name__)


class DingTalkBotHandler(dingtalk_stream.ChatbotHandler):
    work_dir: str
    chat_client: LLMProvider
    registry: Registry
    prompt_composer: PromptComposer
    enable_thinking: bool

    def __init__(
            self,
            work_dir: str,
            chat_client: LLMProvider,
            registry: Registry,
            prompt_composer: PromptComposer,
            enable_thinking: bool,
    ):
        super().__init__()
        self.work_dir = work_dir
        self.chat_client = chat_client
        self.registry = registry
        self.prompt_composer = prompt_composer
        self.enable_thinking = enable_thinking

    async def process(self, callback: CallbackMessage):
        incoming_message = dingtalk_stream.ChatbotMessage.from_dict(callback.data)

        # 群聊/私聊 id
        conversation_id = incoming_message.conversation_id
        # 群聊/私聊 类型: '1' 单聊，'2' 群聊
        conversation_type = incoming_message.conversation_type
        # 群聊/私聊 标题
        conversation_title = incoming_message.conversation_title
        # 消息类型: text / picture / richText
        message_type = incoming_message.message_type
        # 消息内容
        content_strip = incoming_message.text.content.strip()
        # 发送人id
        sender_id = incoming_message.sender_staff_id
        # 发送人昵称
        sender_nick = incoming_message.sender_nick

        # 定义一个 reporter
        chatbot_handler = self

        class DingTalkBotReporter:
            def session_start(self) -> None:
                chatbot_handler.reply_text("开始执行任务...", incoming_message)

            def session_end(self) -> None:
                chatbot_handler.reply_text("任务结束...", incoming_message)

            def step_start(self, step_count: int) -> None:
                chatbot_handler.reply_text(f"========== [Turn {step_count}] 开始 ==========", incoming_message)

            def step_end(self, step_count: int) -> None:
                chatbot_handler.reply_text(f"========== [Turn {step_count}] 结束 ==========", incoming_message)

            def on_thinking(self, message: Message) -> None:
                chatbot_handler.reply_text(
                    f"思考结果: {message.content}",
                    incoming_message,
                )

            def pre_tool_call(self, tool_call: ToolCall) -> None:
                chatbot_handler.reply_text(
                    f"🛠️ 执行工具: {tool_call.name}, 参数: {tool_call.arguments}",
                    incoming_message,
                )

            def post_tool_call(self, tool_result: ToolResult) -> None:
                pass

            def on_message(self, message: Message) -> None:
                chatbot_handler.reply_text(
                    f"🤖 [对外回复]: {message.content}",
                    incoming_message,
                )

        # 发起一个 loop
        # 实例化一个 engine
        agent_engine = AgentEngine(
            provider=self.chat_client,
            registry=self.registry,
            prompt_composer=self.prompt_composer,
            reporter=DingTalkBotReporter(),
            work_dir=self.work_dir,
            enable_thinking=self.enable_thinking,
        )
        try:
            await agent_engine.run(content_strip)
        except:
            logger.exception("loop 运行失败")
            return AckMessage.STATUS_SYSTEM_EXCEPTION, 'error'
        return AckMessage.STATUS_OK, 'ok'


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
    work_dir = os.getenv("WORK_DIR", os.getcwd())
    # 获取绝对路径
    work_dir = str(pathlib.Path(work_dir).expanduser().resolve())

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

    # 提示词组合器
    prompt_composer = PromptComposer(work_dir=work_dir)

    # 钉钉客户端
    client_id = os.getenv("DINGTALK_CLIENT_ID", None)
    client_secret = os.getenv("DINGTALK_CLIENT_SECRET", None)
    if client_id is None or client_secret is None:
        raise ValueError("请设置 DINGTALK_CLIENT_ID 和 DINGTALK_CLIENT_SECRET 环境变量")
    dingtalk_stream_client = DingTalkStreamClient(Credential(client_id, client_secret))
    dingtalk_stream_client.register_callback_handler(
        dingtalk_stream.chatbot.ChatbotMessage.TOPIC,
        DingTalkBotHandler(
            work_dir=work_dir,
            chat_client=chat_client,
            registry=registry,
            prompt_composer=prompt_composer,
            enable_thinking=os.getenv("ENABLE_THINKING", False),
        )
    )

    return DingTalkBot(
        dingtalk_stream_client=dingtalk_stream_client,
        work_dir=work_dir,
    )
