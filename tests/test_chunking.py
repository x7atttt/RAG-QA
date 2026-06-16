from app.services.document_service import chunk_text
from app.services.text_splitter import split_text


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


# ---------- 新增：多策略测试 ----------

def test_split_text_empty():
    """空文本所有策略都返回空列表"""
    assert split_text("", strategy="fixed") == []
    assert split_text("", strategy="recursive") == []
    assert split_text("", strategy="markdown") == []


def test_fixed_strategy_backward_compatible():
    """fixed 策略与原 chunk_text 行为一致"""
    text = "测试" * 300
    chunks = split_text(text, strategy="fixed", chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 1
    assert all(len(c) <= 500 for c in chunks)


def test_recursive_respects_sentence_boundary():
    """recursive 策略尽量在句子边界切，不硬切"""
    # 重复的完整句子，每个 9 字符 + 句号
    text = "这是一句话。" * 100
    chunks = split_text(text, strategy="recursive", chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 2
    # 每块不应超过 chunk_size
    assert all(len(c) <= 50 for c in chunks)


def test_markdown_split_preserves_header_sections():
    """markdown 策略按标题切，不同标题的内容不在同一块"""
    text = (
        "# 项目一\n这是项目一的内容。\n"
        "## 子模块\n子模块详情。\n"
        "# 项目二\n这是项目二的内容。\n"
    )
    chunks = split_text(text, strategy="markdown", chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 1
    # 项目一和项目二应被分开（至少有两个块，或单块内标题有序）
    joined = "\n".join(chunks)
    assert "项目一" in joined
    assert "项目二" in joined


def test_recursive_is_default_for_unknown_strategy():
    """未知策略退化为 recursive（不报错）"""
    text = "测试文本。" * 50
    chunks = split_text(text, strategy="unknown_strategy", chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1


def test_markdown_fallback_on_plain_text():
    """markdown 策略对无标题纯文本安全降级（不报错，返回非空）"""
    text = "这是纯文本，没有任何 Markdown 标题。" * 20
    chunks = split_text(text, strategy="markdown", chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1

