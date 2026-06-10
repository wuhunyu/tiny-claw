import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from src.config.config import settings
from src.core.context import Context
from src.excetion.exceptions import InvalidParamException
from src.schema.message import ToolDefinition

logger = logging.getLogger(__name__)


class WebSearchParams(BaseModel):
    query: str = Field(..., description="搜索内容")
    max_results: int = Field(default=5, description="最大结果数")
    start_date: str = Field(default=datetime.now().strftime("%Y-%m-%d"), description="开始时间(yyyy-MM-dd)")
    end_date: str = Field(default=datetime.now().strftime("%Y-%m-%d"), description="结束时间(yyyy-MM-dd)")


class WebSearchByTavily:
    tavily_client: AsyncTavilyClient

    def __init__(
            self,
            tavily_api_key: str | None = None,
    ):
        tavily_api_key = tavily_api_key or settings.tavily_api_key
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    def name(self) -> str:
        return "web_search_by_tavily"

    def readonly(self) -> bool:
        return True

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description="网络搜索 by Tavily",
            readonly=self.readonly(),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索内容"
                    },
                    "max_results": {
                        "type": ["integer", "null"],
                        "description": "最大结果数"
                    },
                    "start_date": {
                        "type": ["string", "null"],
                        "format": "date",
                        "description": "开始时间(yyyy-MM-dd)"
                    },
                    "end_date": {
                        "type": ["string", "null"],
                        "format": "date",
                        "description": "结束时间(yyyy-MM-dd)"
                    }
                },
                "required": ["query", "max_results", "start_date", "end_date"]
            }
        )

    async def execute(self, context: Context, arguments: dict[str, Any] | str | None) -> str:
        if not arguments:
            logger.info("请指定 query")
            raise InvalidParamException(message="请指定 query")
        if isinstance(arguments, dict):
            web_search_params = WebSearchParams.model_validate(arguments)
        elif isinstance(arguments, str):
            web_search_params = WebSearchParams.model_validate_json(arguments)
        else:
            logger.info(f"{self.name()} 格式错误")
            raise InvalidParamException(message=f"{self.name()} 格式错误")

        resp = await self.tavily_client.search(
            query=web_search_params.query,
            max_results=web_search_params.max_results,
            start_date=web_search_params.start_date,
            end_date=web_search_params.end_date,
        )
        ans = []
        for result in resp.get("results", []):
            title = result.get("title", None)
            url = result.get("url", None)
            content = result.get("content", None)
            if title is None or url is None or content is None:
                continue
            ans.append(f"""
标题: {title}
来源: {url}
详细内容: {content}
""")
        return "\n".join(ans) if ans else "暂无搜索结果"
