import pytest

from app.services.embedding_service import (
    encode_query_full,
    encode_single,
    encode_texts,
    sparse_score,
)


@pytest.mark.asyncio
async def test_encode_single_dim():
    vec = await encode_single("测试文本")
    assert isinstance(vec, list)
    assert len(vec) == 1024


@pytest.mark.asyncio
async def test_encode_batch_dim():
    vecs = await encode_texts(["文本一", "文本二", "文本三"])
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 1024


@pytest.mark.asyncio
async def test_encode_empty_list():
    assert await encode_texts([]) == []


@pytest.mark.asyncio
async def test_encode_chinese_english():
    vecs = await encode_texts([
        "使用 FastAPI 搭建 RESTful API",
        "The endpoint accepts POST requests",
        "纯中文测试",
    ])
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 1024


# ---------------------------------------------------------------------------
# 稀疏检索（sparse / lexical_weights）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_query_full_returns_dense_and_sparse():
    """encode_query_full 应同时返回 dense 向量与 sparse lexical_weights。"""
    enc = await encode_query_full("YOLO 的损失函数")
    assert isinstance(enc["dense"], list)
    assert len(enc["dense"]) == 1024
    # sparse 是 {token_id: weight} 字典，非空（有意义的查询必有稀疏词项）
    assert isinstance(enc["sparse"], dict)
    assert len(enc["sparse"]) > 0
    # key 是 int token_id，value 是 float 权重
    tid, w = next(iter(enc["sparse"].items()))
    assert isinstance(tid, int)
    assert isinstance(w, float)


@pytest.mark.asyncio
async def test_sparse_score_empty():
    assert await sparse_score({}, []) == []


@pytest.mark.asyncio
async def test_sparse_score_ranks_keyword_match_higher():
    """含精确术语的文档 sparse 得分应显著高于无关文档。

    这是 sparse 路的价值：对关键词/术语的字面匹配强于 dense 的语义模糊匹配。
    """
    qe = await encode_query_full("Transformer 自注意力机制 self-attention")
    docs = [
        "Transformer 采用自注意力机制（self-attention）计算序列依赖",
        "今天天气不错适合户外运动",
    ]
    scores = await sparse_score(qe["sparse"], docs)
    assert len(scores) == 2
    # 相关文档得分应远高于无关文档（至少 5 倍以上）
    assert scores[0] > scores[1] * 5
