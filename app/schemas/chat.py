from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # 是否开启 DeepSeek thinking 模式（用户自定义，默认关闭）
    thinking: bool = False
    # 指定会话继续对话；为空则后端自动新建会话
    conversation_id: int | None = None


class SourceItem(BaseModel):
    document_id: int | None = None
    filename: str = ""
    chunk_index: int = 0
    content: str = ""
    score: float = 0.0


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    sources: list[SourceItem] = []
    reasoning: str | None = None
    created_at: datetime | None = None


class ChatHistoryData(BaseModel):
    messages: list[MessageOut]
    next_cursor: int | None = None
    has_next: bool = False


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ConversationListData(BaseModel):
    conversations: list[ConversationOut]
    total: int
