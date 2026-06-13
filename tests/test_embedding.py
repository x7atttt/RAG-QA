import pytest

from app.services.embedding_service import encode_single, encode_texts


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
