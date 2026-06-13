from app.services.document_service import chunk_text


def test_chunk_basic():
    text = "a" * 700
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_overlap():
    text = "abcdefgh" * 100
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    if len(chunks) >= 2:
        assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_empty():
    assert chunk_text("", 500, 100) == []
    assert chunk_text("   \n\n  ", 500, 100) == []


def test_chunk_short_text():
    chunks = chunk_text("短文本", chunk_size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == "短文本"
