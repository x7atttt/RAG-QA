"""CRAG（Corrective RAG）循环重查测试。

验证 transform_query 节点 + retrieve_documents 对变体的支持：
1. transform_query 正常生成变体（mock LLM）
2. transform_query LLM 异常时复用原 query（降级不阻断）
3. retrieve_documents 优先用 transformed_query 检索（优先级正确）

通过 mock app.agent.nodes.chat 控制 LLM 返回，不依赖真实 API。
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.agent.nodes import retrieve_documents, transform_query
from app.agent.state import AgentState


def _make_state(
    question="测试问题",
    rewritten_query=None,
    transformed_query=None,
    retrieved_docs=None,
) -> AgentState:
    return {
        "user_id": 1,
        "question": question,
        "rewritten_query": rewritten_query or "",
        "transformed_query": transformed_query or "",
        "retrieval_attempts": 0,
        "history": [],
        "should_retrieve": True,
        "retrieved_docs": retrieved_docs or [],
        "sources": [],
    }


# ============ transform_query 节点测试 ============

@pytest.mark.asyncio
async def test_transform_query_generates_variant():
    """有弱命中片段时，LLM 应被调用，生成同义/换角度变体。"""
    state = _make_state(
        question="加密方案",
        rewritten_query="加密方案",
        retrieved_docs=["密钥管理相关片段", "对称加密说明"],
    )

    with patch(
        "app.agent.nodes.chat",
        new_callable=AsyncMock,
        return_value="密钥管理与加密实现方式",
    ):
        result = await transform_query(state)

    assert result["transformed_query"] == "密钥管理与加密实现方式"
    assert result["question"] == "加密方案"  # 原始问题不变


@pytest.mark.asyncio
async def test_transform_query_strips_quotes_and_punctuation():
    """变体应去除首尾引号/句号（LLM 有时会用引号包裹，含中文弯引号）。"""
    state = _make_state(question="备份策略", retrieved_docs=["容灾片段"])

    # LLM 返回带中文弯引号和句号
    curved = "\u201c\u6570\u636e\u5907\u4efd\u4e0e\u5bb9\u707e\u65b9\u6848\u201d\u3002"
    with patch(
        "app.agent.nodes.chat",
        new_callable=AsyncMock,
        return_value=curved,
    ):
        result = await transform_query(state)

    assert result["transformed_query"] == "数据备份与容灾方案"


@pytest.mark.asyncio
async def test_transform_query_llm_exception_falls_back_to_base():
    """LLM 异常时复用原 query（不阻断重查流程）。"""
    state = _make_state(
        question="审计日志",
        rewritten_query="审计日志",  # 有 rewrite 结果作基线
        retrieved_docs=["日志片段"],
    )

    with patch("app.agent.nodes.chat", new_callable=AsyncMock, side_effect=Exception("API 超时")):
        result = await transform_query(state)

    # 降级：复用 rewritten_query，不抛异常
    assert result["transformed_query"] == "审计日志"


@pytest.mark.asyncio
async def test_transform_query_empty_response_keeps_base():
    """LLM 返回空时复用原 query。"""
    state = _make_state(question="安全规范", retrieved_docs=["规范片段"])

    with patch("app.agent.nodes.chat", new_callable=AsyncMock, return_value="   "):
        result = await transform_query(state)

    assert result["transformed_query"] == "安全规范"


# ============ retrieve_documents 优先级测试 ============

@pytest.mark.asyncio
async def test_retrieve_uses_transformed_query_when_present():
    """transformed_query 非空时，retrieve 应优先用它检索（而非 rewritten_query/question）。"""
    state = _make_state(
        question="原始问题",
        rewritten_query="指代消解后的问题",
        transformed_query="CRAG变体重写后的问题",
    )

    # mock 编码，验证 retrieve 内部用的 search_query 是 transformed_query
    captured_query = {}

    async def fake_encode(query):
        captured_query["value"] = query
        return {"dense": [0.1] * 8, "sparse": {}}

    with (
        patch("app.agent.nodes.encode_query_full", new_callable=AsyncMock, side_effect=fake_encode),
        patch("app.agent.nodes.get_user_collection") as mock_col,
    ):
        mock_col.return_value.count.return_value = 1
        mock_col.return_value.query.return_value = {"documents": [[]], "metadatas": [[]]}
        await retrieve_documents(state)

    # encode_query_full 收到的 query 应是 transformed_query（CRAG 变体优先级最高）
    assert captured_query.get("value") == "CRAG变体重写后的问题"


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_rewritten_query_without_transform():
    """无 transformed_query 时，retrieve 用 rewritten_query。"""
    state = _make_state(
        question="原始问题",
        rewritten_query="指代消解后的问题",
        transformed_query="",  # 无 CRAG 变体
    )

    captured_query = {}

    async def fake_encode(query):
        captured_query["value"] = query
        return {"dense": [0.1] * 8, "sparse": {}}

    with (
        patch("app.agent.nodes.encode_query_full", new_callable=AsyncMock, side_effect=fake_encode),
        patch("app.agent.nodes.get_user_collection") as mock_col,
    ):
        mock_col.return_value.count.return_value = 1
        mock_col.return_value.query.return_value = {"documents": [[]], "metadatas": [[]]}
        await retrieve_documents(state)

    assert captured_query.get("value") == "指代消解后的问题"


# ============ grade_documents 评分节点测试 ============

def test_grade_high_score_no_retry():
    """top score ≥ 阈值 → 不触发重查（should_retry=False）。"""
    from app.agent.nodes import grade_documents

    state = _make_state()
    state["sources"] = [{"score": 0.89}, {"score": 0.7}]
    result = grade_documents(state)
    assert result["meta"]["should_retry"] is False


def test_grade_low_score_triggers_retry():
    """top score < 阈值 且未达上限 → 触发重查（should_retry=True, attempts+1）。"""
    from app.agent.nodes import grade_documents

    state = _make_state()
    state["sources"] = [{"score": 0.3}]  # 低于阈值 0.5
    state["retrieval_attempts"] = 0
    result = grade_documents(state)
    assert result["meta"]["should_retry"] is True
    assert result["retrieval_attempts"] == 1  # 计数 +1


def test_grade_max_attempts_no_retry():
    """已达重查上限 → 不再重查（即使 score 低）。"""
    from app.agent.nodes import grade_documents
    from app.config import get_settings

    max_attempts = get_settings().crag_max_attempts  # 默认 1
    state = _make_state()
    state["sources"] = [{"score": 0.2}]  # 仍然低
    state["retrieval_attempts"] = max_attempts  # 已达上限
    result = grade_documents(state)
    assert result["meta"]["should_retry"] is False  # 不再重查，走 fallback


def test_grade_empty_sources_no_retry():
    """无召回结果 → 不重查（没有内容可改写，直接走 fallback）。"""
    from app.agent.nodes import grade_documents

    state = _make_state()
    state["sources"] = []
    result = grade_documents(state)
    assert result["meta"]["should_retry"] is False
