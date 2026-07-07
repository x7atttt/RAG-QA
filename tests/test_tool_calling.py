"""工具调用测试。

覆盖 spec/20260706-tool-calling.md 的第一版工具：
1. tool_router：LLM 决策是否调文档元信息工具
2. doc_meta：文档元信息查询（用户隔离 + done 状态过滤）
3. web_search：缺 Tavily key 时显式失败，由调用方降级
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from langchain_core.messages import HumanMessage

from app.agent.nodes import tool_router, _is_explicit_web_search_command
from app.agent.state import AgentState
from app.services.chat_service import stream_graph
from app.services.tools.doc_meta import format_for_prompt, query_doc_meta
from app.services.tools.web_search import WebSearchError, web_search


def _make_state(question="我上传过哪些 PDF？") -> AgentState:
    return {
        "user_id": 1,
        "question": question,
        "history": [],
        "summary": "",
        "should_retrieve": True,
        "tool_call": None,
    }


@pytest.mark.asyncio
async def test_explicit_web_search_command_sets_web_search_tool_call():
    state = _make_state("用联网搜索查一下2026年欧洲杯赛程")
    out = await tool_router(state)
    assert out["tool_call"] == "web_search"


@pytest.mark.asyncio
async def test_tool_router_routes_doc_meta_list():
    state = _make_state("我上传过哪些 PDF？")

    with patch("app.agent.nodes.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"call": true, "intent": "list"}'
        out = await tool_router(state)

    assert out["tool_call"] == "doc_meta"
    assert out["doc_meta_intent"] == "list"



@pytest.mark.asyncio
async def test_tool_router_routes_doc_meta_recent():
    state = _make_state("最近一周上传了哪些文档？")

    with patch("app.agent.nodes.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"call": true, "intent": "recent"}'
        out = await tool_router(state)

    assert out["tool_call"] == "doc_meta"
    assert out["doc_meta_intent"] == "recent"


@pytest.mark.asyncio
async def test_tool_router_no_tool_for_content_question():
    state = _make_state("文档里讲了什么？")

    with patch("app.agent.nodes.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"call": false}'
        out = await tool_router(state)

    assert out["tool_call"] is None


@pytest.mark.asyncio
async def test_tool_router_invalid_json_falls_back_to_no_tool():
    state = _make_state()

    with patch("app.agent.nodes.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "不是 JSON"
        out = await tool_router(state)

    assert out["tool_call"] is None


@pytest.mark.asyncio
async def test_doc_meta_query_filters_user_and_status(client):
    from app.core.database import async_session_factory
    from app.models.document import Document

    async with async_session_factory() as db:
        db.add_all([
            Document(user_id=1, filename="a.pdf", file_type="pdf", chunk_count=3, file_size=1024, status="done"),
            Document(user_id=1, filename="b.md", file_type="md", chunk_count=1, file_size=512, status="pending"),
            Document(user_id=2, filename="other.pdf", file_type="pdf", chunk_count=2, file_size=2048, status="done"),
        ])
        await db.commit()

        items = await query_doc_meta(db, user_id=1, intent="list")

    assert [item["filename"] for item in items] == ["a.pdf"]


@pytest.mark.asyncio
async def test_doc_meta_format_for_prompt():
    text = format_for_prompt([
        {
            "filename": "a.pdf",
            "file_type": "pdf",
            "chunk_count": 3,
            "file_size": 2048,
            "status": "done",
            "created_at": "2026-07-06T10:00:00",
        }
    ], "list")

    assert "a.pdf" in text
    assert "pdf" in text
    assert "3 块" in text


@pytest.mark.asyncio
async def test_web_search_without_key_raises_error():
    fake_settings = SimpleNamespace(tavily_api_key="", tavily_max_results=3)

    with patch("app.services.tools.web_search.get_settings", return_value=fake_settings):
        with pytest.raises(WebSearchError):
            await web_search("测试 query")


async def _fake_astream_chat(messages, thinking=False):
    yield ("content", "ok")


@pytest.mark.asyncio
async def test_stream_graph_doc_meta_branch_skips_retrieval():
    async def fake_intent(state):
        state["should_retrieve"] = True
        return state

    async def fake_tool_router(state):
        state["tool_call"] = "doc_meta"
        state["doc_meta_intent"] = "list"
        return state

    async def fake_run_doc_meta(state):
        return [HumanMessage(content="文档元信息")]

    with (
        patch("app.agent.nodes.intent_router", new=fake_intent),
        patch("app.agent.nodes.tool_router", new=fake_tool_router),
        patch("app.services.chat_service._run_doc_meta", new=fake_run_doc_meta) as mock_run,
        patch("app.agent.nodes.retrieve_documents", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.llm_provider.astream_chat", new=_fake_astream_chat),
    ):
        events = [event async for event in stream_graph(None, 1, "我上传过哪些文档？")]

    assert ("token", "ok") in events
    assert not mock_retrieve.called


@pytest.mark.asyncio
async def test_stream_graph_low_score_uses_web_search_when_enabled():
    async def fake_intent(state):
        state["should_retrieve"] = True
        return state

    async def fake_tool_router(state):
        state["tool_call"] = None
        return state

    async def fake_rewrite(state):
        state["rewritten_query"] = state["question"]
        return state

    async def fake_retrieve(state):
        state["retrieved_docs"] = ["弱相关片段"]
        state["sources"] = [{"document_id": 1, "filename": "a.md", "chunk_index": 0, "content": "弱相关片段", "score": 0.1}]
        return state

    async def fake_try_web_search(state, enable_web_search=True):
        return "[1] 网络结果\n来源：https://example.com\n内容"

    with (
        patch("app.agent.nodes.intent_router", new=fake_intent),
        patch("app.agent.nodes.tool_router", new=fake_tool_router),
        patch("app.agent.nodes.rewrite_query", new=fake_rewrite),
        patch("app.agent.nodes.retrieve_documents", new=fake_retrieve),
        patch("app.services.chat_service._try_web_search", new=fake_try_web_search),
        patch("app.services.chat_service.settings.crag_max_attempts", 0),
        patch("app.services.llm_provider.astream_chat", new=_fake_astream_chat),
    ):
        events = [event async for event in stream_graph(None, 1, "外部问题")]

    assert ("sources", [{"document_id": 1, "filename": "a.md", "chunk_index": 0, "content": "弱相关片段", "score": 0.1}]) in events
    assert ("token", "ok") in events
    assert any(ev == "web_search" for ev, _ in events)


@pytest.mark.asyncio
async def test_stream_graph_explicit_command_triggers_web_search():
    async def fake_tool_router(state):
        if "联网搜索" in (state.get("question") or ""):
            state["tool_call"] = "web_search"
        else:
            state["tool_call"] = None
        return state

    async def fake_try_web_search(state, enable_web_search=True):
        return "[1] 命中联网结果\n来源：https://example.com\n内容"

    with (
        patch("app.agent.nodes.tool_router", new=fake_tool_router),
        patch("app.services.chat_service._try_web_search", new=fake_try_web_search),
        patch("app.agent.nodes.intent_router", new_callable=AsyncMock) as mock_intent,
        patch("app.agent.nodes.retrieve_documents", new_callable=AsyncMock) as mock_retrieve,
        patch("app.services.llm_provider.astream_chat", new=_fake_astream_chat),
    ):
        events = [event async for event in stream_graph(None, 1, "用联网搜索查一下2026年欧洲杯赛程")]

    assert any(ev == "web_search" and payload.get("used") is True for ev, payload in events if isinstance(payload, dict))
    assert ("token", "ok") in events
    assert not mock_intent.called
    assert not mock_retrieve.called
