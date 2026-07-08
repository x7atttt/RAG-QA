import json
from collections.abc import AsyncIterator
from typing import Any

from app.agent.state import AgentState
from app.config import get_settings
from app.core.database import async_session_factory

settings = get_settings()


def sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


# 手动编排节点执行（替代原 graph.astream_events 方案）：
# 原 LangChain ChatDeepSeek 的 on_chat_model_stream 事件在改用 OpenAI SDK 后不再触发，
# 故改为 stream_graph 直接调用节点函数，在生成阶段用 astream_chat 实时 yield token/reasoning。
# 这样流式推送完全可控，thinking 模式的 reasoning_content 能正确推送到前端。


async def stream_graph(
    graph,
    user_id: int,
    question: str,
    history: list[dict] | None = None,
    thinking: bool = False,
    summary: str | None = None,
    enable_web_search: bool = True,
) -> AsyncIterator[tuple[str, Any]]:
    """yield (event_name, payload). event ∈ reasoning/token/sources/done/error.

    手动编排（不走 graph.astream_events），编排顺序（spec/20260706-tool-calling.md）：
    1. intent_router → 判断要不要检索（内容层）
    2. tool_router → 判断是不是元信息问题（元层），是则调 doc_meta 工具跳过检索
    3. 若需检索：rewrite_query + CRAG 循环重查 → 推 sources
    4. CRAG 仍失败 + 开启联网搜索 → 调 web_search 兜底
    5. 选 prompt（rag / fallback / web_search / doc_meta / 纯对话）→ astream_chat 流式生成

    放弃 LangGraph 的 astream_events：它依赖 LangChain 的 on_chat_model_stream 事件，
    改用 OpenAI SDK 直连后该事件不触发。手动编排让生成阶段的流式完全可控，
    thinking 模式的 reasoning_content 实时推送，让前端"深度思考"开关真正生效。
    """
    # 延迟导入避免循环依赖（nodes 依赖 llm_provider，不依赖本模块）
    from app.agent.nodes import (
        _build_doc_meta_prompt,
        _build_fallback_prompt,
        _build_rag_prompt,
        _build_web_search_prompt,
        _history_to_messages,
        _inject_summary,
        intent_router,
        retrieve_documents,
        rewrite_query,
        tool_router,
        transform_query,
    )
    from app.services.llm_provider import astream_chat
    from langchain_core.messages import HumanMessage

    initial: AgentState = {
        "user_id": user_id,
        "question": question,
        "rewritten_query": "",
        "transformed_query": "",
        "retrieval_attempts": 0,
        "history": history or [],
        "summary": summary or "",
        "thinking": thinking,
        "should_retrieve": False,
        "tool_call": None,
        "doc_meta_intent": "",
        "doc_meta_result": "",
        "web_search_result": "",
        "retrieved_docs": [],
        "sources": [],
        "answer_tokens": [],
        "answer": "",
        "reasoning_tokens": [],
        "reasoning": "",
        "error": None,
        "meta": {},
    }

    answer_parts: list[str] = []
    reasoning_parts: list[str] = []

    try:
        state = initial

        # 1. 工具路由（显式命令优先：如“用联网搜索查一下”）
        state = await tool_router(state)

        # 2. 意图路由（内容层：要不要检索文档内容）
        if state.get("tool_call") != "web_search":
            state = await intent_router(state)

        # 3. 显式联网搜索 → 跳过检索链路，直接触发联网搜索
        if state.get("tool_call") == "web_search":
            state["should_retrieve"] = False
            web_context = await _try_web_search(state, enable_web_search=enable_web_search)
            if web_context:
                yield ("web_search", {"used": True, "query": state["question"], "preview": web_context[:300]})
                sources = state.get("sources") or []
                sources = [*sources, {"filename": "联网搜索结果", "source": "web"}]
                state["sources"] = sources
                yield ("sources", sources)
                messages = _build_web_search_prompt(
                    state["question"], web_context, fallback_docs=[],
                    history=state.get("history", []), summary=state.get("summary") or None,
                )
            else:
                yield ("web_search", {"used": False})
                messages = _history_to_messages(state.get("history", []))
                messages = _inject_summary(messages, state.get("summary") or None)
                messages.append(HumanMessage(content=state["question"]))

        # 分支 A：文档元信息工具 → 跳过检索，直接走 doc_meta 生成
        elif state.get("tool_call") == "doc_meta":
            messages = await _run_doc_meta(state)

        # 分支 B：检索链路（含 CRAG + 联网搜索兜底）
        elif state.get("should_retrieve"):
            state = await rewrite_query(state)
            state["retrieval_attempts"] = 0
            # CRAG 循环：检索后若 top score 低于阈值且未达重查上限 →
            # transform_query 改写变体 → 重新 retrieve（静默，用户无感）
            while True:
                state = await retrieve_documents(state)
                attempts = state.get("retrieval_attempts", 0)
                top_score = state["sources"][0].get("score", 0) if state.get("sources") else 0
                if (
                    state.get("sources")
                    and top_score < settings.retrieval_score_threshold
                    and attempts < settings.crag_max_attempts
                ):
                    state["retrieval_attempts"] = attempts + 1
                    state = await transform_query(state)
                    continue  # 用变体重查
                break
            # 构造生成消息：高相关走 RAG；低相关尝试联网搜索兜底，否则 fallback
            question_text = state["question"]
            history_list = state.get("history", [])
            summary_text = state.get("summary") or None
            docs = state.get("retrieved_docs", [])
            sources = state.get("sources", [])
            top_score = sources[0].get("score", 0) if sources else 0

            if docs and top_score >= settings.retrieval_score_threshold:
                # 高相关：推送完整 sources 给前端 + 存 DB
                if sources:
                    yield ("sources", sources)
                messages = _build_rag_prompt(question_text, docs, history_list, summary_text)
            else:
                web_context = await _try_web_search(state, enable_web_search=enable_web_search)
                if web_context:
                    # 联网搜索触发时，给前端可观测事件 + 历史来源标记
                    yield ("web_search", {"used": True, "query": state.get("transformed_query") or state.get("rewritten_query") or question_text, "preview": web_context[:300]})
                    sources = [{"filename": "联网搜索结果", "source": "web"}]
                    state["sources"] = sources
                    yield ("sources", sources)
                    messages = _build_web_search_prompt(
                        question_text, web_context, fallback_docs=docs,
                        history=history_list, summary=summary_text,
                    )
                else:
                    # fallback：低分 sources 不推送（回答说"文档未涉及"，附来源自相矛盾）
                    yield ("web_search", {"used": False})
                    state["sources"] = []
                    yield ("sources", [])
                    messages = _build_fallback_prompt(question_text, docs, history_list, summary_text)

        # 分支 C：纯对话（general 路径）：摘要也注入，保持长对话连贯
        else:
            state["retrieved_docs"] = []
            state["sources"] = []
            history_list = state.get("history", [])
            summary_text = state.get("summary") or None
            messages = _history_to_messages(history_list)
            messages = _inject_summary(messages, summary_text)
            messages.append(HumanMessage(content=question))

        # 3. 流式生成：直接用 OpenAI SDK 的 astream_chat，实时 yield
        async for event, text in astream_chat(messages, thinking=thinking):
            if event == "reasoning":
                reasoning_parts.append(text)
                yield ("reasoning", text)
            elif event == "content":
                answer_parts.append(text)
                yield ("token", text)

        yield ("answer_final", {
            "answer": "".join(answer_parts),
            "reasoning": "".join(reasoning_parts),
        })
        yield ("done", {"status": "ok"})
    except Exception as e:
        yield ("error", {"message": str(e)})


async def _run_doc_meta(state: AgentState) -> list:
    """执行文档元信息工具并构造生成消息。

    独立成函数：工具调用 + DB 查询 + prompt 构造的内聚封装。
    DB session 在此处按需开（不触发改工具分支时不开），不影响其他分支。
    """
    from app.agent.nodes import _build_doc_meta_prompt
    from app.services.tools.doc_meta import format_for_prompt, query_doc_meta

    user_id = state["user_id"]
    intent = state.get("doc_meta_intent") or "list"
    # 安全转换：intent 由 LLM 产出，可能非法
    intent = intent if intent in ("list", "recent") else "list"

    async with async_session_factory() as db:
        items = await query_doc_meta(db, user_id, intent=intent)

    state["doc_meta_result"] = format_for_prompt(items, intent)
    return _build_doc_meta_prompt(
        state["question"], state["doc_meta_result"],
        history=state.get("history"), summary=state.get("summary") or None,
    )


async def _try_web_search(state: AgentState, enable_web_search: bool = True) -> str:
    """规则触发联网搜索（CRAG 失败兜底，或显式联网搜索命令触发）。

    返回 web_context 文本：非空 = 搜到了结果，调用方走 _build_web_search_prompt；
    空 = 跳过（未开启 / 缺 KEY / 调用失败 / 无结果），调用方降级走 _build_fallback_prompt。
    任何异常都降级为空字符串，不阻断主流程。
    """
    if not enable_web_search:
        return ""
    from app.services.tools.web_search import WebSearchError, format_for_prompt, web_search

    # 检索用 query：CRAG 变体 > 指代消解后的 query > 原始 question（与 retrieve 一致）
    search_query = (
        state.get("transformed_query")
        or state.get("rewritten_query")
        or state["question"]
    )
    try:
        results = await web_search(search_query)
    except WebSearchError:
        return ""
    web_context = format_for_prompt(results)
    state["web_search_result"] = web_context
    return web_context
