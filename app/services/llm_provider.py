"""LLM Provider 层：用 OpenAI SDK 直连 DeepSeek（兼容 OpenAI 格式）。

替代原先的 langchain-deepseek.ChatDeepSeek。改用 OpenAI SDK 的原因：
1. langchain-deepseek 绑死 DeepSeek，多模型切换成本高
2. thinking 开关/reasoning_content 的处理更可控（官方文档示例就是 OpenAI SDK）
3. 减少一个依赖层（langchain-deepseek 只是搬运 reasoning_content，自己取 3 行代码搞定）

设计：
- AsyncOpenAI 客户端单例（进程级共享连接池）
- chat()：非流式（intent_router / rewrite_query 用），temperature=0 确定性输出
- astream_chat()：流式生成器，yield ("reasoning"|"content", text) 元组
  - thinking=True 时 reasoning_content 先于 content 返回（前端推理面板先展示思考过程）
  - reasoning_content 从 delta.reasoning_content 取（DeepSeek 在 OpenAI 返回结构上的扩展字段）
- thinking 开关通过 extra_body 透传（DeepSeek 官方推荐方式）
- 思考模式不支持 temperature 等参数（官方文档），故 thinking 时强制不传
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("docqa.llm")

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """获取 AsyncOpenAI 客户端单例（进程级共享连接池）。"""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    return _client


def _to_openai_messages(messages: list) -> list[dict[str, str]]:
    """把 LangChain 消息对象或 dict 转成 OpenAI SDK 格式。

    支持 LangChain 的 SystemMessage/HumanMessage/AIMessage（有 .type/.content）
    和原生 dict（{"role":..., "content":...}）。

    注意：LangChain 的 .type 用 "human"/"ai" 等，OpenAI 要 "user"/"assistant"，
    需要做 role 映射。
    """
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            # LangChain 消息对象：HumanMessage.type='human', SystemMessage.type='system'
            langchain_role = getattr(m, "type", None) or "user"
            role = role_map.get(langchain_role, langchain_role)
            content = getattr(m, "content", "") or ""
        out.append({"role": role, "content": content})
    return out


async def chat(
    messages: list,
    *,
    thinking: bool = False,
    max_tokens: int = 512,
) -> str:
    """非流式对话（intent_router / rewrite_query 用）。

    temperature=0 保证确定性输出（intent 判断 / query 改写需要稳定结果）。
    thinking 模式下不传 temperature（DeepSeek 思考模式不支持，传了也不生效）。
    """
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": _to_openai_messages(messages),
        "max_tokens": max_tokens,
    }
    if thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        # 显式禁用思考模式（deepseek-v4-flash 默认 enabled）+ temperature=0 确定性输出
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        kwargs["temperature"] = 0
    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


async def astream_chat(
    messages: list,
    *,
    thinking: bool = False,
    max_tokens: int = 1024,
) -> AsyncIterator[tuple[str, str]]:
    """流式生成器，yield (event, text)。

    event ∈ {"reasoning", "content"}：
    - thinking=True 时，reasoning_content（思维链）先于 content（最终答案）返回
    - thinking=False 时，只 yield content（DeepSeek 可能仍返回少量 reasoning，
      但不影响：前端只在 thinking 开关开启时展示推理面板）

    reasoning_content 通过 getattr(delta, "reasoning_content", "") 取——
    DeepSeek 在 OpenAI 返回结构里扩展了这个字段，OpenAI SDK 透传未知字段，
    需要用 getattr 防 AttributeError（其他模型如 GPT 无此字段）。
    """
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": _to_openai_messages(messages),
        "max_tokens": max_tokens,
        "stream": True,
    }
    if thinking:
        # 思考模式：思考模式不支持 temperature 等参数（官方文档），故不传
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        # 显式禁用思考模式：deepseek-v4-flash 默认 thinking=enabled，
        # 不显式 disabled 会导致 reasoning_content 始终输出，前端开关形同虚设。
        # 思考模式关闭后才支持 temperature（确定性输出，便于评测复现）
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        kwargs["temperature"] = 0

    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # reasoning_content（思维链）——DeepSeek 扩展字段，用 getattr 兜底
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ("reasoning", reasoning)
        # content（最终答案）
        content = delta.content
        if content:
            yield ("content", content)
