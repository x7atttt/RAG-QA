from typing import Any, TypedDict


class SourceItem(TypedDict):
    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float


class AgentState(TypedDict, total=False):
    user_id: int
    question: str
    should_retrieve: bool
    retrieved_docs: list[str]
    sources: list[SourceItem]
    answer_tokens: list[str]
    answer: str
    error: str | None
    meta: dict[str, Any]
