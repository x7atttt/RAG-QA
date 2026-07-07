"""文档元信息查询工具。

定位（spec/20260706-tool-calling.md 决策 3）：
- 触发方式 = LLM 决策（tool_router 判断问题类型为"元信息查询"时调用）
- 工具本身是纯数据查询（不接 LLM）——LLM 决策在 tool_router 节点，工具只执行
- 解决的问题：检索是 chunk 级语义匹配，答不了"我有哪些文档""最近加了什么"这类元层问题

支持两类查询（intent 参数由 tool_router 的 LLM 提取）：
- list：列出文档（可按 file_type 过滤，可按时间排序）
- recent：最近 N 天上传的文档

返回文档元信息列表，由调用方（stream_graph）拼成 user message 喂给 LLM 生成自然语言回答。
"""

import logging
from datetime import datetime, timedelta
from typing import Literal, TypedDict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document

logger = logging.getLogger("docqa.tools.doc_meta")


class DocMetaItem(TypedDict):
    """文档元信息条目（仅暴露对用户有意义的字段）。"""

    filename: str
    file_type: str
    chunk_count: int
    file_size: int
    status: str
    created_at: str  # ISO 格式字符串


DocMetaIntent = Literal["list", "recent"]


async def query_doc_meta(
    db: AsyncSession,
    user_id: int,
    intent: DocMetaIntent,
    file_type: str | None = None,
    recent_days: int = 7,
    limit: int = 20,
) -> list[DocMetaItem]:
    """查询用户文档元信息。

    Args:
        db: 数据库会话（由调用方提供，复用其事务边界）
        user_id: 用户 ID（数据隔离）
        intent: 查询意图——list（列出文档）/ recent（最近 N 天）
        file_type: 文档类型过滤（如 "pdf"），None 表示不过滤
        recent_days: intent=recent 时的天数窗口
        limit: 返回条数上限（防止超长输出撑爆 prompt）
    """
    stmt = select(Document).where(
        Document.user_id == user_id,
        Document.status == "done",  # 只看处理完成的，pending/failed 对用户无意义
    )

    if intent == "recent":
        since = datetime.now() - timedelta(days=recent_days)
        stmt = stmt.where(Document.created_at >= since).order_by(Document.created_at.desc())
    else:  # list
        if file_type:
            stmt = stmt.where(Document.file_type == file_type)
        stmt = stmt.order_by(Document.created_at.desc())

    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    docs = result.scalars().all()

    items = [
        DocMetaItem(
            filename=d.filename,
            file_type=d.file_type,
            chunk_count=d.chunk_count,
            file_size=d.file_size,
            status=d.status,
            created_at=d.created_at.isoformat() if d.created_at else "",
        )
        for d in docs
    ]
    logger.info(f"doc_meta 查询（user={user_id}, intent={intent}）返回 {len(items)} 条")
    return items


def format_for_prompt(items: list[DocMetaItem], intent: DocMetaIntent) -> str:
    """把元信息列表格式化为可拼入 prompt 的文本块。"""
    if not items:
        return "（无符合条件的文档）"

    lines = [f"共 {len(items)} 份文档："]
    for d in items:
        size_kb = d["file_size"] / 1024
        lines.append(
            f"- {d['filename']}（{d['file_type']}，{d['chunk_count']} 块，"
            f"{size_kb:.1f}KB，上传于 {d['created_at'][:10]}）"
        )
    return "\n".join(lines)
