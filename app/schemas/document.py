from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: str
    chunk_count: int
    file_size: int
    created_at: datetime | None = None


class DocumentListData(BaseModel):
    documents: list[DocumentOut]
    next_cursor: int | None = None
    has_next: bool = False
