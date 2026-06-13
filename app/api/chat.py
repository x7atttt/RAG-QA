import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import async_session_factory, get_db
from app.core.exceptions import BizError
from app.core.response import ResponseCode, success_response
from app.models import Conversation, Message, User
from app.schemas.chat import ChatAskRequest, ChatHistoryData, MessageOut, SourceItem
from app.services.chat_service import sse, stream_graph

router = APIRouter()


@router.post("/ask")
async def ask(
    request: Request,
    body: ChatAskRequest,
    user: User = Depends(get_current_user),
):
    graph = request.app.state.graph
    question = body.question.strip()
    if not question:
        raise BizError(code=ResponseCode.EMPTY_QUESTION, message="问题不能为空", http_status=400)

    user_id = user.id

    async def event_stream():
        answer_parts: list[str] = []
        sources: list[dict] = []
        error_msg: str | None = None

        async for event_name, payload in stream_graph(graph, user_id, question):
            yield sse(event_name, payload)
            if event_name == "token":
                answer_parts.append(payload)
            elif event_name == "sources":
                sources = payload
            elif event_name == "error":
                error_msg = payload.get("message", "未知错误")

        if error_msg is None and answer_parts:
            answer = "".join(answer_parts)
            async with async_session_factory() as db:
                conv = Conversation(user_id=user_id, title=question[:50])
                db.add(conv)
                await db.flush()
                conv_id = conv.id

                db.add(Message(conversation_id=conv_id, role="user", content=question, sources=None))
                db.add(
                    Message(
                        conversation_id=conv_id,
                        role="assistant",
                        content=answer,
                        sources=json.dumps(sources, ensure_ascii=False) if sources else None,
                    )
                )
                await db.commit()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history")
async def history(
    cursor: int | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = (
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Conversation.user_id == user.id)
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
                created_at=m.created_at,
            )
        )

    data = ChatHistoryData(messages=out, next_cursor=next_cursor, has_next=has_next)
    return success_response(data.model_dump(mode="json"))
