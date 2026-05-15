import logging
import pathlib

import aiofiles

from src.context.skill import SkillLoader
from src.schema.message import Message, Role, OutputSchema

logger = logging.getLogger(__name__)


class PromptComposer:
    AGENTS_FILE_NAME = "AGENTS.md"

    work_dir: str
    skill_loader: SkillLoader

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.skill_loader = SkillLoader(work_dir=work_dir)

    async def build(self) -> Message:
        prompts = [
            # 基础约束
            """
# 核心身份
你名叫 go-tiny-claw，一个由驾驭工程驱动的骨灰级研发助手。
你具备极简主义哲学，拒绝废话。
你能通过系统提供的内置工具，创建、读取、修改和执行工作区中的代码。

# 核心纪律 (CRITICAL)
1. 如需检查文件是否存在，请使用 bash 的 ls 或 test -f，而不是对目录使用 read_file。
2. 创建新文件时，务必使用 write_file，并同时提供 path 和 content 参数。
3. 编辑文件前务必先读取现有文件，以理解上下文。
4. 无论何时你需要写代码或创建文件，都要直接使用 write_file 工具。
5. 遇到工具执行报错时，仔细阅读 stderr，尝试自己修正命令并重试。
6. 始终用中文回复，以便传达你的进展和想法。
            """,
            # 补充输出格式要求
            OutputSchema.PROMPT,
        ]

        # 加载 AGENTS.md
        # 获取工作空间路径的绝对路径
        absolute_work_dir = pathlib.Path(self.work_dir).expanduser().resolve()
        agents_file = absolute_work_dir / self.AGENTS_FILE_NAME
        if agents_file.is_file():
            try:
                async with aiofiles.open(agents_file, "r") as f:
                    agents_content = await f.read()
            except IOError:
                logger.warning(f"无法读取 {self.AGENTS_FILE_NAME} 文件", exc_info=True)
                agents_content = ""
            if agents_content:
                prompts.append(f"""

# 项目专属指南 (来自 AGENTS.md)
以下是当前工作区特有的架构规范与注意事项，你的行为必须绝对符合以下要求：
```markdown
{agents_content}
```

                """)

        # 加载 skills
        skills_prompt = self.skill_loader.load_all()
        if skills_prompt:
            prompts.append(skills_prompt)

        return Message(
            role=Role.ROLE_SYSTEM,
            content="\n".join(prompts),
        )
