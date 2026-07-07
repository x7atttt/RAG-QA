"""对话中断与继续回答测试。

验证：
1. _finalize(is_partial=True) 正确标记 partial
2. /api/chat/continue 端点校验与流程
3. history API 返回 status 字段
"""
import json
from unittest.mock import patch

import pytest


async def _register_and_login(client, username="cont_user", password="secret123"):
    await client.post("/api/auth/register", json={"username": username, "password": password})
    resp = await client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.json()["data"]["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _create_conversation(client, token: str) -> int:
    resp = await client.post("/api/chat/conversations", headers=_auth(token))
    return resp.json()["data"]["id"]


async def _create_partial_message(client, token: str, conversation_id: int) -> int:
    """直接写入一条 partial 消息到 DB，模拟用户中断。"""
    from app.core.database import async_session_factory
    from app.models import Message

    async with async_session_factory() as db:
        # 先插入 user 消息
        user_msg = Message(conversation_id=conversation_id, role="user", content="测试问题", status="complete")
        db.add(user_msg)
        # 插入 partial assistant 消息
        asst_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content="这是部分答案...",
            sources=json.dumps([{"filename": "test.pdf", "score": 0.9}]),
            status="partial",
        )
        db.add(asst_msg)
        await db.commit()
        await db.refresh(asst_msg)
        return asst_msg.id


# ---------- partial 保存 ----------


@pytest.mark.asyncio
async def test_finalize_saves_partial(client):
    """_finalize(is_partial=True) 将消息标记为 partial。"""
    token = await _register_and_login(client)
    conv_id = await _create_conversation(client, token)

    from app.api.chat import _finalize

    await _finalize(
        user_id=1,
        question="test q",
        answer_parts=["partial answer"],
        reasoning_parts=[],
        sources=[],
        error_msg=None,
        lock_token=None,
        conversation_id=conv_id,
        is_partial=True,
    )

    from app.core.database import async_session_factory
    from app.models import Message
    from sqlalchemy import select

    async with async_session_factory() as db:
        msg = (await db.execute(
            select(Message).where(Message.conversation_id == conv_id, Message.role == "assistant")
        )).scalar_one()
    assert msg.content == "partial answer"
    assert msg.status == "partial"


@pytest.mark.asyncio
async def test_finalize_saves_complete(client):
    """_finalize(is_partial=False) 将消息标记为 complete（默认）。"""
    token = await _register_and_login(client, "cont_user2")
    conv_id = await _create_conversation(client, token)

    from app.api.chat import _finalize

    await _finalize(
        user_id=1,
        question="test q",
        answer_parts=["complete answer"],
        reasoning_parts=[],
        sources=[],
        error_msg=None,
        lock_token=None,
        conversation_id=conv_id,
        is_partial=False,
    )

    from app.core.database import async_session_factory
    from app.models import Message
    from sqlalchemy import select

    async with async_session_factory() as db:
        msg = (await db.execute(
            select(Message).where(Message.conversation_id == conv_id, Message.role == "assistant")
        )).scalar_one()
    assert msg.content == "complete answer"
    assert msg.status == "complete"


# ---------- history API ----------


@pytest.mark.asyncio
async def test_history_returns_status_field(client):
    """history API 返回 status 字段。"""
    token = await _register_and_login(client, "cont_user3")
    conv_id = await _create_conversation(client, token)
    await _create_partial_message(client, token, conv_id)

    resp = await client.get(f"/api/chat/history?conversation_id={conv_id}", headers=_auth(token))
    assert resp.status_code == 200
    msgs = resp.json()["data"]["messages"]
    asst_msgs = [m for m in msgs if m["role"] == "assistant"]
    assert len(asst_msgs) == 1
    assert asst_msgs[0]["status"] == "partial"


# ---------- /continue 端点 ----------


@pytest.mark.asyncio
async def test_continue_rejects_non_partial(client):
    """对 complete 消息调 continue 应被拒绝。"""
    token = await _register_and_login(client, "cont_user4")
    conv_id = await _create_conversation(client, token)

    # 创建 complete 消息
    from app.core.database import async_session_factory
    from app.models import Message

    async with async_session_factory() as db:
        user_msg = Message(conversation_id=conv_id, role="user", content="q", status="complete")
        db.add(user_msg)
        asst_msg = Message(conversation_id=conv_id, role="assistant", content="full answer", status="complete")
        db.add(asst_msg)
        await db.commit()
        await db.refresh(asst_msg)
        msg_id = asst_msg.id

    resp = await client.post(
        "/api/chat/continue",
        json={"message_id": msg_id, "conversation_id": conv_id},
        headers=_auth(token),
    )
    assert resp.status_code == 400
    assert "非中断状态" in resp.json()["message"]


@pytest.mark.asyncio
async def test_continue_rejects_wrong_message_id(client):
    """message_id 不存在时应 404。"""
    token = await _register_and_login(client, "cont_user5")
    conv_id = await _create_conversation(client, token)

    resp = await client.post(
        "/api/chat/continue",
        json={"message_id": 99999, "conversation_id": conv_id},
        headers=_auth(token),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_continue_rejects_wrong_conversation(client):
    """message_id 不属于指定 conversation 时应 404。"""
    token = await _register_and_login(client, "cont_user6")
    conv_id = await _create_conversation(client, token)
    other_conv_id = await _create_conversation(client, token)
    msg_id = await _create_partial_message(client, token, conv_id)

    resp = await client.post(
        "/api/chat/continue",
        json={"message_id": msg_id, "conversation_id": other_conv_id},
        headers=_auth(token),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_continue_streams_and_merges(client):
    """continue 端点流式续写并合并到原 message。"""
    token = await _register_and_login(client, "cont_user7")
    conv_id = await _create_conversation(client, token)
    msg_id = await _create_partial_message(client, token, conv_id)

    async def _fake_astream_chat(messages, thinking=False):
        yield ("content", "续写内容")

    with patch("app.services.llm_provider.astream_chat", side_effect=_fake_astream_chat):
        resp = await client.post(
            "/api/chat/continue",
            json={"message_id": msg_id, "conversation_id": conv_id},
            headers=_auth(token),
        )
    assert resp.status_code == 200
    body = resp.text
    assert "event: token" in body or "event: done" in body or "event: answer_final" in body

    # 验证消息已合并为 complete
    from app.core.database import async_session_factory
    from app.models import Message

    async with async_session_factory() as db:
        msg = await db.get(Message, msg_id)
    assert msg.status == "complete"
    # 内容应该是部分答案 + 续写内容
    assert "这是部分答案..." in msg.content
    assert "续写内容" in msg.content


@pytest.mark.asyncio
async def test_continue_rejects_wrong_user(client):
    """其他用户的 token 调 continue 应被拒绝。"""
    token_a = await _register_and_login(client, "owner_user")
    token_b = await _register_and_login(client, "other_user")
    conv_id = await _create_conversation(client, token_a)
    msg_id = await _create_partial_message(client, token_a, conv_id)

    resp = await client.post(
        "/api/chat/continue",
        json={"message_id": msg_id, "conversation_id": conv_id},
        headers=_auth(token_b),
    )
    assert resp.status_code == 404
