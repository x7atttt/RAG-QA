from datetime import datetime

from sqlalchemy import String, Integer, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    # 文件内容 sha256，用于用户内去重（联合唯一索引见 database.py）
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # 文档处理状态：pending(已入库待处理) / processing(解析中) / done(完成) / failed(失败)
    # 默认 done 兼容旧数据（init_db 给旧库补列时 status 为 NULL，应用层兜底视为 done）
    status: Mapped[str] = mapped_column(String(20), default="done", server_default="done")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
