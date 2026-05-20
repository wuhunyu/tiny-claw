import os
import pathlib

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.provider.interface import Provider


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    work_dir: str = Field(default_factory=os.getcwd, alias="WORK_DIR", description="工作空间目录")

    dingtalk_client_id: str = Field(..., alias="DINGTALK_CLIENT_ID", description="钉钉客户端ID", min_length=1)
    dingtalk_client_secret: str = Field(..., alias="DINGTALK_CLIENT_SECRET", description="钉钉客户端密钥", min_length=1)

    llm_provider: Provider = Field(default=Provider.OPENAI, alias="LLM_PROVIDER", description="LLM 提供商")
    llm_model: str = Field(default="gpt-5.3-codex", alias="LLM_MODEL", description="LLM 模型")
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="LLM_BASE_URL",
        description="LLM BASE URL",
    )
    llm_api_key: str = Field(..., alias="LLM_API_KEY", description="LLM API Key", min_length=1)

    enable_thinking: bool = Field(default=False, alias="ENABLE_THINKING", description="是否启用思考模式")

    bash_timeout: int = Field(default=30, alias="BASH_TIMEOUT", description="Bash 命令执行超时时间", ge=1)

    max_session_window_size: int = Field(default=10, alias="MAX_SESSION_WINDOW_SIZE", description="会话窗口大小", ge=1)

    tavily_api_key: str = Field(..., alias="TAVILY_API_KEY", description="Tavily API Key", min_length=1)

    @field_validator("work_dir")
    @classmethod
    def normalize_work_dir(cls, value: str) -> str:
        return str(pathlib.Path(value).expanduser().resolve())


settings = AppSettings()
