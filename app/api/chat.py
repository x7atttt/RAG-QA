import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import get_settings
from app.core.cache import (
    acquire_lock,
    get_cached_answer,
    get_history_cache,
    invalidate_history_cache,
    release_lock,
    set_cached_answer,
    set_history_cache,
)
from app.core.database import async_session_factory, get_db
from app.core.exceptions import BizError
from app.core.rate_limit import limiter
from app.core.response import ResponseCode, success_response
from app.models import Conversation, Message, User
from app.schemas.chat import (
    ChatAskRequest,
    ChatContinueRequest,
    ChatHistoryData,
    ConversationListData,
    ConversationOut,
    MessageOut,
    SourceItem,
)
from app.services.chat_service import sse, stream_graph

router = APIRouter()
settings = get_settings()


def _stream_cached(cached: dict, cache_tag: str):
    async def gen():
        if cached.get("sources"):
            yield sse("sources", cached["sources"])
        # 缓存命中时回放完整 reasoning（如有）
        reasoning = cached.get("reasoning")
        if reasoning:
            yield sse("reasoning", reasoning)
        if cached.get("answer"):
            for i in range(0, len(cached["answer"]), 4):
                yield sse("token", cached["answer"][i : i + 4])
        yield sse("done", {"status": "ok", "cache": cache_tag})

    return gen()


@router.post("/ask")
@limiter.limit("100/minute")
async def ask(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ChatAskRequest,
    user: User = Depends(get_current_user),
):
    graph = request.app.state.graph
    question = body.question.strip()
    if not question:
        raise BizError(code=ResponseCode.EMPTY_QUESTION, message="问题不能为空", http_status=400)

    user_id = user.id
    thinking = bool(body.thinking)
    conversation_id = body.conversation_id

    # 取当前会话最近若干轮历史作为多轮上下文（正序：最旧在前）
    # 实际窗口由节点层按 token 预算 + 轮数截断
    history = await _load_recent_history(user_id, conversation_id)
    # 加载会话摘要（长对话老上下文压缩），注入 system prompt 做长期记忆
    summary = await _load_summary(conversation_id)

    # 缓存按会话隔离（key 含 conversation_id）
    # history_version：用历史消息条数做上下文版本，避免多轮追问时命中上一轮上下文的过期答案
    hver = len(history)
    hit, cached = await get_cached_answer(user_id, question, conversation_id, hver)
    if hit and cached and cached.get("answer") and bool(cached.get("thinking", False)) == thinking:
        return StreamingResponse(
            _stream_cached(cached, "hit"),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    lock_token: str | None = await acquire_lock(user_id, question, conversation_id, history_version=hver)

    if lock_token is None:
        for _ in range(8):
            await asyncio.sleep(0.3)
            hit2, cached2 = await get_cached_answer(user_id, question, conversation_id, hver)
            if hit2 and cached2 and cached2.get("answer") and bool(cached2.get("thinking", False)) == thinking:
                return StreamingResponse(
                    _stream_cached(cached2, "wait"),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
        lock_token = await acquire_lock(user_id, question, conversation_id, history_version=hver)
        if lock_token is None:
            lock_token = "no-redis"

    async def event_stream():
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        sources: list[dict] = []
        error_msg: str | None = None
        partial = False

        try:
            async for event_name, payload in stream_graph(
                graph, user_id, question, history, thinking, summary, enable_web_search=bool(getattr(body, 'enable_web_search', True))
            ):
                yield sse(event_name, payload)
                if event_name == "token":
                    answer_parts.append(payload)
                elif event_name == "reasoning":
                    reasoning_parts.append(payload)
                elif event_name == "sources":
                    sources = payload
                elif event_name == "answer_final":
                    # 用流末的权威完整答案/推理覆盖（避免流式拼接遗漏）
                    if isinstance(payload, dict):
                        if payload.get("answer"):
                            answer_parts = [payload["answer"]]
                        if payload.get("reasoning"):
                            reasoning_parts = [payload["reasoning"]]
                elif event_name == "error":
                    error_msg = payload.get("message", "未知错误")
        except asyncio.CancelledError:
            partial = True
            raise
        finally:
            # partial 时必须同步等待落库（shield 在 generator 退出后任务会被取消）；
            # 完整回答用 shield 异步不阻塞。
            coro = _finalize(
                user_id=user_id,
                question=question,
                answer_parts=answer_parts,
                reasoning_parts=reasoning_parts,
                sources=sources,
                error_msg=error_msg,
                lock_token=lock_token,
                background_tasks=background_tasks,
                thinking=thinking,
                conversation_id=conversation_id,
                history_version=hver,
                is_partial=partial,
            )
            if partial:
                await coro
            else:
                await asyncio.shield(coro)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/continue")
@limiter.limit("100/minute")
async def continue_answer(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ChatContinueRequest,
    user: User = Depends(get_current_user),
):
    """继续回答被中断的 partial 消息。

    流程：加载 partial 消息 → 构造续写 prompt → astream_chat 流式续写 → UPDATE 合并。
    """
    graph = request.app.state.graph

    # 1. 加载 partial 消息，校验归属
    async with async_session_factory() as db:
        msg = await db.get(Message, body.message_id)
        if not msg or msg.conversation_id != body.conversation_id:
            raise BizError(code=ResponseCode.CONVERSATION_NOT_FOUND, message="消息不存在", http_status=404)
        conv = await db.get(Conversation, body.conversation_id)
        if not conv or conv.user_id != user.id:
            raise BizError(code=ResponseCode.CONVERSATION_NOT_FOUND, message="会话不存在", http_status=404)
        if msg.status != "partial":
            raise BizError(code=ResponseCode.BAD_REQUEST, message="该消息非中断状态，无需继续", http_status=400)
        partial_answer = msg.content
        partial_reasoning = msg.reasoning or ""
        # 从 sources JSON 反推原始来源类型
        raw_sources = msg.sources
        existing_sources: list[dict] = json.loads(raw_sources) if raw_sources else []

    # 2. 取历史 + 摘要（与 ask 一致）
    user_id = user.id
    conversation_id = body.conversation_id
    thinking = bool(body.thinking)
    history = await _load_recent_history(user_id, conversation_id)
    summary = await _load_summary(conversation_id)

    # 3. 找到原始问题（partial 消息的前一条 user 消息）
    async with async_session_factory() as db:
        prev_msg = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id, Message.role == "user", Message.id < body.message_id)
            .order_by(Message.id.desc())
            .limit(1)
        )
        user_msg = prev_msg.scalar_one_or_none()
    if not user_msg:
        raise BizError(code=ResponseCode.BAD_REQUEST, message="找不到对应的问题", http_status=400)
    question = user_msg.content

    # 4. 构造续写 prompt：history + 问题 + 部分答案 + 续写指令
    from app.agent.nodes import _history_to_messages, _inject_summary
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    system = (
        "你是文档问答助手。请继续回答用户的问题。\n"
        "要求：1) 从上一条回答的断点处无缝续写，不要重复已有内容；\n"
        "2) 保持风格和语气一致；3) 简洁专业，中文回答。"
    )
    messages = [SystemMessage(content=system)]
    if history:
        messages.extend(_history_to_messages(history))
    messages = _inject_summary(messages, summary)
    messages.append(HumanMessage(content=question))
    messages.append(AIMessage(content=partial_answer))
    messages.append(HumanMessage(content="请继续上面的回答，从断点处无缝续写，不要重复已有内容。"))

    # 5. 流式续写 + 实时推送
    from app.services.llm_provider import astream_chat

    cont_answer_parts: list[str] = []
    cont_reasoning_parts: list[str] = []

    async def continue_stream():
        nonlocal cont_answer_parts, cont_reasoning_parts
        partial_inner = False
        try:
            async for event, text in astream_chat(messages, thinking=thinking):
                if event == "reasoning":
                    cont_reasoning_parts.append(text)
                    yield sse("reasoning", text)
                elif event == "content":
                    cont_answer_parts.append(text)
                    yield sse("token", text)
            # 续写完成：合并到原 message
            full_answer = partial_answer + "".join(cont_answer_parts)
            full_reasoning = partial_reasoning + "".join(cont_reasoning_parts) if (partial_reasoning or cont_reasoning_parts) else ""
            await _merge_continuation(
                message_id=body.message_id,
                conversation_id=conversation_id,
                full_answer=full_answer,
                full_reasoning=full_reasoning,
                sources=existing_sources,
                background_tasks=background_tasks,
            )
            yield sse("answer_final", {"answer": full_answer, "reasoning": full_reasoning})
            yield sse("done", {"status": "ok"})
        except asyncio.CancelledError:
            partial_inner = True
            # 再次中断：合并已续写部分，仍标记 partial
            if cont_answer_parts:
                merged = partial_answer + "".join(cont_answer_parts)
                merged_reasoning = partial_reasoning + "".join(cont_reasoning_parts) if cont_reasoning_parts else partial_reasoning
                await _merge_continuation(
                    message_id=body.message_id,
                    conversation_id=conversation_id,
                    full_answer=merged,
                    full_reasoning=merged_reasoning,
                    sources=existing_sources,
                    background_tasks=background_tasks,
                    is_partial=True,
                )
            raise
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(
        continue_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _merge_continuation(
    message_id: int,
    conversation_id: int,
    full_answer: str,
    full_reasoning: str,
    sources: list[dict],
    background_tasks: BackgroundTasks | None = None,
    is_partial: bool = False,
) -> None:
    """续写完成后 UPDATE 原 message：合并内容，status → complete（或保持 partial）。"""
    try:
        async with async_session_factory() as db:
            msg = await db.get(Message, message_id)
            if msg:
                msg.content = full_answer
                msg.reasoning = full_reasoning or None
                msg.sources = json.dumps(sources, ensure_ascii=False) if sources else None
                msg.status = "partial" if is_partial else "complete"
                await db.commit()
            await invalidate_history_cache(conversation_id)
            if background_tasks is not None and not is_partial:
                from app.services.summary_service import maybe_generate_summary
                background_tasks.add_task(maybe_generate_summary, conversation_id)
    except Exception:
        pass


async def _load_recent_history(
    user_id: int,
    conversation_id: int | None,
    rounds: int | None = None,
) -> list[dict]:
    """读取指定会话最近 N 轮历史消息，返回正序（最旧在前）。

    默认取 summarize_round_threshold 轮（比生成窗口更宽），交由节点层按
    token 预算 + 轮数截断——既满足生成时的窗口控制，也为会话摘要预留
    看到窗口外老对话的能力。conversation_id 为 None 时取全局最近消息（兼容旧调用）。

    读路径：有 conversation_id 时先查 Redis 历史缓存，miss 再查 DB 回填。
    """
    if rounds is None:
        rounds = settings.summarize_round_threshold

    # 有会话 id：缓存优先（按会话维度缓存）
    if conversation_id is not None:
        cached = await get_history_cache(conversation_id)
        if cached is not None:
            return cached

    history = await _load_history_from_db(user_id, conversation_id, rounds)

    # 回填缓存（仅会话维度可缓存）
    if conversation_id is not None and history:
        await set_history_cache(conversation_id, history)
    return history


async def _load_history_from_db(
    user_id: int,
    conversation_id: int | None,
    rounds: int,
) -> list[dict]:
    """从 DB 查询历史消息（缓存未命中时调用）。"""
    try:
        async with async_session_factory() as db:
            stmt = (
                select(Message)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(Conversation.user_id == user_id)
                .order_by(Message.id.desc())
                .limit(rounds * 2)
            )
            if conversation_id is not None:
                stmt = stmt.where(Message.conversation_id == conversation_id)
            result = await db.execute(stmt)
            msgs = result.scalars().all()
        return [
            {"role": m.role, "content": m.content}
            for m in reversed(msgs)
            if m.role in ("user", "assistant") and m.content
        ]
    except Exception:
        return []


async def _load_summary(conversation_id: int | None) -> str | None:
    """读取会话摘要（长对话老上下文压缩）。无会话或未生成摘要返回 None。"""
    if conversation_id is None:
        return None
    try:
        async with async_session_factory() as db:
            conv = await db.get(Conversation, conversation_id)
            return conv.summary if conv else None
    except Exception:
        return None


async def _finalize(
    user_id: int,
    question: str,
    answer_parts: list[str],
    reasoning_parts: list[str] | None,
    sources: list[dict],
    error_msg: str | None,
    lock_token: str | None,
    background_tasks: BackgroundTasks | None = None,
    thinking: bool = False,
    conversation_id: int | None = None,
    history_version: int = 0,
    is_partial: bool = False,
) -> None:
    answer = "".join(answer_parts)
    reasoning = "".join(reasoning_parts) if reasoning_parts else ""
    if lock_token:
        await release_lock(user_id, question, lock_token, conversation_id, history_version)
    if error_msg or not answer:
        return
    try:
        await set_cached_answer(user_id, question, answer, sources, conversation_id, reasoning, thinking, history_version)
    except Exception:
        pass
    try:
        async with async_session_factory() as db:
            # 复用指定会话（校验 user_id 防越权），否则新建
            conv_id: int
            if conversation_id is not None:
                conv = await db.get(Conversation, conversation_id)
                if conv and conv.user_id == user_id:
                    # 首问时把"新对话"更新为问题摘要
                    if conv.title == "新对话":
                        conv.title = question[:20]
                    conv_id = conv.id
                else:
                    # 会话不存在或越权 → 新建
                    conv = Conversation(user_id=user_id, title=question[:20])
                    db.add(conv)
                    await db.flush()
                    conv_id = conv.id
            else:
                conv = Conversation(user_id=user_id, title=question[:20])
                db.add(conv)
                await db.flush()
                conv_id = conv.id
            db.add(
                Message(conversation_id=conv_id, role="user", content=question, sources=None)
            )
            db.add(
                Message(
                    conversation_id=conv_id,
                    role="assistant",
                    content=answer,
                    sources=json.dumps(sources, ensure_ascii=False) if sources else None,
                    reasoning=reasoning or None,
                    status="partial" if is_partial else "complete",
                )
            )
            await db.commit()
            # 新消息落地 → 失效历史缓存（避免下次读到不含本次问答的脏历史）
            await invalidate_history_cache(conv_id)
            # 消息落库后，异步判断是否需要生成/刷新会话摘要（达阈值才真正调 LLM）
            # 放进 BackgroundTasks：用户已收到流式回答，摘要生成不阻塞请求
            if background_tasks is not None:
                from app.services.summary_service import maybe_generate_summary

                background_tasks.add_task(maybe_generate_summary, conv_id)
    except Exception:
        pass


@router.get("/history")
async def history(
    conversation_id: int = Query(..., description="会话 ID"),
    cursor: int | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # 校验会话归属当前用户
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user.id:
        raise BizError(code=ResponseCode.CONVERSATION_NOT_FOUND, message="会话不存在", http_status=404)

    query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
    )
    if cursor is not None:
        query = query.where(Message.id < cursor)
    query = query.limit(limit + 1)

    result = await db.execute(query)
    messages = result.scalars().all()
    has_next = len(messages) > limit
    messages = messages[:limit]
    next_cursor = messages[-1].id if has_next and messages else None

    out: list[MessageOut] = []
    for m in messages:
        sources_list: list[SourceItem] = []
        if m.sources:
            try:
                raw = json.loads(m.sources)
                sources_list = [SourceItem(**s) for s in raw]
            except Exception:
                sources_list = []
        out.append(
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=sources_list,
                reasoning=m.reasoning,
                status=m.status or "complete",
                created_at=m.created_at,
            )
        )

    data = ChatHistoryData(messages=out, next_cursor=next_cursor, has_next=has_next)
    return success_response(data.model_dump(mode="json"))


# ---------- 会话管理 ----------


@router.get("/conversations")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """列出当前用户所有会话，按 updated_at 倒序。"""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    convs = result.scalars().all()
    out = [ConversationOut.model_validate(c) for c in convs]
    data = ConversationListData(conversations=out, total=len(out))
    return success_response(data.model_dump(mode="json"))


@router.post("/conversations")
async def create_conversation(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """新建空会话。达上限则拒绝创建。"""
    # 检查上限
    cnt_result = await db.execute(
        select(func.count()).select_from(Conversation).where(Conversation.user_id == user.id)
    )
    count = cnt_result.scalar() or 0
    if count >= settings.max_conversations:
        raise BizError(
            code=ResponseCode.CONVERSATION_LIMIT_EXCEEDED,
            message=f"会话已达上限（{settings.max_conversations} 个），请先删除旧会话",
            http_status=409,
        )
    conv = Conversation(user_id=user.id, title="新对话")
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return success_response(ConversationOut.model_validate(conv).model_dump(mode="json"), "创建成功")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """删除会话及其所有消息（外键无 CASCADE，手动先删 Message）。"""
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user.id:
        raise BizError(code=ResponseCode.CONVERSATION_NOT_FOUND, message="会话不存在", http_status=404)
    # 先删该会话所有消息，再删会话（防孤儿消息）
    await db.execute(delete(Message).where(Message.conversation_id == conversation_id))
    await db.delete(conv)
    await db.commit()
    # 删除会话后清掉历史缓存与摘要锁残留（避免幽灵数据）
    await invalidate_history_cache(conversation_id)
    return success_response(None, "删除成功")
