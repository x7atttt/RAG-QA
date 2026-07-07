"""联网搜索工具（Tavily）。

定位（spec/20260706-tool-calling.md 决策 2）：
- 触发方式 = 规则触发（CRAG 重查仍失败后作为兜底信息源）
- 不走 LLM 决策：CRAG 失败是确定性信号，规则触发保证"该搜必搜"
- 失败降级：网络异常/缺 KEY 时返回空列表，调用方降级走原 fallback，不阻断主流程

Tavily 返回结构（https://docs.tavily.com/documentation/api-reference/endpoint/search）：
{
  "results": [
    {"title": "...", "url": "...", "content": "...（已摘要）", "score": 0.xx},
    ...
  ]
}
content 已是 Tavily 抽取过的片段，无需自己抓网页摘要。
"""

import logging
from typing import TypedDict

import httpx

from app.config import get_settings

logger = logging.getLogger("docqa.tools.web_search")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class WebSearchResult(TypedDict):
    """单条搜索结果。"""

    title: str
    url: str
    content: str


class WebSearchError(Exception):
    """联网搜索失败（缺 KEY / 网络 / API 错误）。"""


async def web_search(query: str, max_results: int | None = None) -> list[WebSearchResult]:
    """调用 Tavily 搜索，返回最多 max_results 条结果。

    失败一律抛 WebSearchError，由调用方降级处理（不让工具异常阻断主流程）。
    """
    settings = get_settings()
    api_key = settings.tavily_api_key

    if not api_key:
        logger.warning("web_search 跳过：未配置 TAVILY_API_KEY")
        raise WebSearchError("TAVILY_API_KEY 未配置")

    max_r = max_results or settings.tavily_max_results

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_r,
        # 已有知识库的检索场景，搜索深度用 basic 即可（advanced 更慢更贵）
        "search_depth": "basic",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(TAVILY_SEARCH_URL, json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"web_search 调用失败：{e}")
        raise WebSearchError(f"Tavily 请求失败：{e}") from e

    data = resp.json()
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        }
        for item in data.get("results", [])
    ]
    logger.info(f"web_search 返回 {len(results)} 条结果（query={query[:30]}...）")
    return results


def format_for_prompt(results: list[WebSearchResult]) -> str:
    """把搜索结果格式化为可拼入 prompt 的文本块。"""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\n来源：{r['url']}\n{r['content']}")
    return "\n\n".join(parts)
