import os.path
import pathlib

import frontmatter
from pydantic import BaseModel, Field


class Skill(BaseModel):
    name: str = Field(..., description="技能名称")
    description: str = Field(..., description="技能描述")
    content: str = Field(..., description="技能内容")
    path: str = Field(..., description="技能路径")


class SkillLoader(BaseModel):
    _SKILL_BASE = os.path.join(".claw", "skills")
    _SKILL_FILE_NAME = "SKILL.md"

    work_dir: str = Field(..., description="工作目录")

    def load_all(self) -> str:
        # 获取工作空间路径的绝对路径
        absolute_work_dir = pathlib.Path(self.work_dir).expanduser().resolve()
        absolute_skill_base = absolute_work_dir / self._SKILL_BASE
        if not absolute_skill_base.exists() or not absolute_skill_base.is_dir():
            return ""

        skills_prompt = [
            "\n"
            "### 可用的专业技能(Agent Skills)"
            "以下是你拥有的标准化外挂技能，请在符合 description 描述的场景下严格遵循其正文指令:"
            "\n"
            "\n"
        ]
        base_skills_prompt_length = len(skills_prompt)
        for skill_path in absolute_skill_base.iterdir():
            if not skill_path.is_dir():
                continue
            skill = self.parse_skill_md(skill_path)
            if not skill:
                continue
            skills_prompt.append(f"#### 技能名称: {skill.name}")
            skills_prompt.append(f"**触发条件**: {skill.description}")
            skills_prompt.append(f"**执行指南**: {skill.content}")
            skills_prompt.append("\n\n")

        return "\n".join(skills_prompt) if len(skills_prompt) > base_skills_prompt_length else ""

    def parse_skill_md(self, skill_path: pathlib.Path) -> Skill | None:
        skill_file = skill_path / self._SKILL_FILE_NAME
        if not skill_file.is_file():
            return None
        try:
            skill_md = frontmatter.load(skill_file)
        except IOError:
            return None
        if not skill_md:
            return None
        skill_name = str(skill_md.metadata.get("name", "")).strip()
        skill_description = str(skill_md.metadata.get("description", "")).strip()
        skill_content = skill_md.content.strip()
        if not skill_name or not skill_description or not skill_content:
            return None
        return Skill(
            name=skill_name,
            description=skill_description,
            content=skill_content,
            path=str(skill_path),
        )
