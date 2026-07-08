from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # 是否开启 DeepSeek thinking 模式（用户自定义，默认关闭）
    thinking: bool = False
    # 指定会话继续对话；为空则后端自动新建会话
    conversation_id: int | None = None
    # 用户级联网搜索开关：默认开启，用户可手动关闭；显式联网搜索命令仍受此开关约束
    enable_web_search: bool = True


class SourceItem(BaseModel):
    document_id: int | None = None
    filename: str = ""
    chunk_index: int = 0
    content: str = ""
    score: float = 0.0
    source: str = ""  # "web" 标记联网搜索结果，前端据此过滤重复 badge


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
