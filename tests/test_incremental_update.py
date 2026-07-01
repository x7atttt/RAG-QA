"""文档增量更新单元测试。

验证 update_document_chunks 的核心行为：
1. 完全无变化 → 不调 encode（全复用）
2. 局部变化（<阈值）→ 只 encode 变化块（增量路径）
3. 大面积变化（>阈值）→ 降级全量重建
4. 新增 + 删除块 → 正确增删

用真实 ChromaDB collection（不 mock 向量库），DB 用 _db fixture。
encode_texts 必须 mock（真实 BGE-M3 太慢且与 diff 逻辑无关）。
"""

from unittest.mock import AsyncMock, patch

import os

import pytest
import pytest_asyncio

from app.core.database import async_session_factory
from app.models import Document
from app.services.document_service import (
    _compute_chunk_hash,
    get_user_collection,
    update_document_chunks,
)


@pytest_asyncio.fixture(scope="function")
async def _db():
    """初始化测试库（建表）+ 测试前后清理 ChromaDB，确保用例间隔离。

    ChromaDB PersistentClient 按 user_id 建集合且全局持久，测试间会残留数据
    互相污染。故每个用例：开始前重置 client 单例 + 删持久化目录，结束后再清。
    """
    import shutil

    from app.core.database import Base, engine, init_db
    from app.services import document_service as ds

    chroma_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_test")

    # 开始前：重置 client 单例 + 清空持久化目录
    ds._chroma_client = None
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir, ignore_errors=True)

    await init_db()
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    # 结束后：再清一次，给下一个用例干净环境
    ds._chroma_client = None
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir, ignore_errors=True)


async def _create_doc_with_chunks(
    user_id: int, filename: str, chunks: list[str], content_hash: bool = True
) -> Document:
    """建文档 + 写入 chunks 到 ChromaDB（带 content_hash），返回 Document。"""
    from app.services.document_service import _build_chunk_metadata
    from app.services.embedding_service import encode_texts

    async with async_session_factory() as db:
        doc = Document(
            user_id=user_id,
            filename=filename,
            file_type="md",
            chunk_count=len(chunks),
            file_size=100,
            file_hash="testhash",
            status="done",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        # 真实写 ChromaDB（带 content_hash metadata）
        collection = get_user_collection(user_id)
        # 用假的 embedding（维度随意，测试不关心向量本身只关心 diff）
        embeddings = [[0.01] * 8 for _ in chunks]
        ids = [f"{doc.id}_chunk_{i}" for i in range(len(chunks))]
        metas = _build_chunk_metadata(user_id, doc.id, filename, chunks) if content_hash else [
            {"user_id": user_id, "document_id": doc.id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metas)
        return doc


def _get_collection_chunk_count(user_id: int, doc_id: int) -> int:
    """查询某文档在 ChromaDB 的当前 chunk 数。"""
    collection = get_user_collection(user_id)
    result = collection.get(where={"document_id": doc_id})
    return len(result["ids"])


@pytest.mark.asyncio
async def test_no_change_skips_encode(_db):
    """新旧分块完全相同 → 不调 encode（全复用）。"""
    chunks = ["第一段内容", "第二段内容", "第三段内容"]
    doc = await _create_doc_with_chunks(95001, "t.md", chunks)

    async with async_session_factory() as db:
        db_doc = await db.get(Document, doc.id)
        with patch(
            "app.services.document_service.encode_texts", new_callable=AsyncMock
        ) as mock_encode:
            stats = await update_document_chunks(db_doc, chunks, 95001, db)

    assert not mock_encode.called  # 没有任何变化，不 encode
    assert stats["reused"] == 3
    assert stats["added"] == 0
    assert stats["removed"] == 0
    assert stats["degraded"] is False


@pytest.mark.asyncio
async def test_partial_change_only_encodes_changed(_db):
    """局部变化（6/8 块未变）→ 只 encode 变化的 2 块（增量路径）。

    8 块改 2 块：union=10 changed=4 ratio=0.4 < 阈值0.5，走增量。
    （集合 diff 下 union 会大于块数，故需足够多未变块确保变化率低于阈值）
    """
    old_chunks = ["块A", "块B", "块C", "块D", "块E", "块F", "块G", "块H"]
    doc = await _create_doc_with_chunks(95002, "t.md", old_chunks)

    # 改 B→B'、G→G'，其余不变
    new_chunks = ["块A", "块B改", "块C", "块D", "块E", "块F", "块G改", "块H"]

    async with async_session_factory() as db:
        db_doc = await db.get(Document, doc.id)
        with patch(
            "app.services.document_service.encode_texts",
            new_callable=AsyncMock,
            return_value=[[0.1] * 8, [0.2] * 8],  # 2 个变化的 embedding
        ) as mock_encode:
            stats = await update_document_chunks(db_doc, new_chunks, 95002, db)

    assert mock_encode.call_count == 1  # 只调一次 encode（2 块）
    encoded = mock_encode.call_args[0][0]
    assert len(encoded) == 2  # 只 encode 2 个变化块
    assert stats["added"] == 2
    assert stats["removed"] == 2
    assert stats["reused"] == 6
    assert stats["degraded"] is False
    # chunk 总数不变
    assert _get_collection_chunk_count(95002, doc.id) == 8


@pytest.mark.asyncio
async def test_high_change_ratio_degrades_to_full(_db):
    """变化率超阈值（4/5 变）→ 降级全量重建。"""
    old_chunks = ["块A", "块B", "块C", "块D", "块E"]
    doc = await _create_doc_with_chunks(95003, "t.md", old_chunks)

    # 5 块里改 4 块，变化率 80% > 50% 阈值
    new_chunks = ["块A", "块B改", "块C改", "块D改", "块E改"]

    async with async_session_factory() as db:
        db_doc = await db.get(Document, doc.id)
        with patch(
            "app.services.document_service.encode_texts",
            new_callable=AsyncMock,
            return_value=[[0.1] * 8] * 5,  # 全量重建 encode 全部 5 块
        ) as mock_encode:
            stats = await update_document_chunks(db_doc, new_chunks, 95003, db)

    assert stats["degraded"] is True
    # 全量重建：一次 encode 全部 5 块（不是增量的小批）
    encoded = mock_encode.call_args[0][0]
    assert len(encoded) == 5
    assert _get_collection_chunk_count(95003, doc.id) == 5


@pytest.mark.asyncio
async def test_added_and_removed_chunks(_db):
    """新增 + 删除块：旧 3 块 → 新 3 块（删1 加1 改1 保1）。"""
    old_chunks = ["块A", "块B", "块C"]
    doc = await _create_doc_with_chunks(95004, "t.md", old_chunks)

    # 删C、改B→B'、加D、保A：变化 2/3 ≈ 67% > 50%... 会降级
    # 调整：让变化率低于阈值。删C、加D、保A、保B：新增1 删除1，变化率 2/4=50%，不超阈值
    new_chunks = ["块A", "块B", "块D"]

    async with async_session_factory() as db:
        db_doc = await db.get(Document, doc.id)
        with patch(
            "app.services.document_service.encode_texts",
            new_callable=AsyncMock,
            return_value=[[0.1] * 8],  # 只新增 1 块（块D）
        ) as mock_encode:
            stats = await update_document_chunks(db_doc, new_chunks, 95004, db)

    assert stats["degraded"] is False
    assert stats["added"] == 1   # 块D 新增
    assert stats["removed"] == 1  # 块C 删除
    assert stats["reused"] == 2   # 块A、块B 复用
    assert _get_collection_chunk_count(95004, doc.id) == 3


@pytest.mark.asyncio
async def test_old_data_without_content_hash_degrades(_db):
    """旧数据无 content_hash → 变化率必然超阈值 → 自动降级全量。

    场景：stage-14 之前上传的文档，metadata 没 content_hash 字段。
    首次更新应降级全量重建，重建后补全 hash，后续增量生效。
    """
    chunks = ["旧块A", "旧块B"]
    # content_hash=False 模拟旧数据
    doc = await _create_doc_with_chunks(95005, "t.md", chunks, content_hash=False)

    # 即使新旧内容完全相同，因旧数据无 hash，removed 含全部旧块 → 降级
    async with async_session_factory() as db:
        db_doc = await db.get(Document, doc.id)
        with patch(
            "app.services.document_service.encode_texts",
            new_callable=AsyncMock,
            return_value=[[0.1] * 8] * 2,
        ) as mock_encode:
            stats = await update_document_chunks(db_doc, chunks, 95005, db)

    assert stats["degraded"] is True  # 旧数据无 hash → 降级全量
    # 全量重建后 metadata 应有 content_hash 了
    collection = get_user_collection(95005)
    result = collection.get(where={"document_id": doc.id})
    assert all("content_hash" in m for m in result["metadatas"])
