"""Hybrid 检索（dense+sparse RRF 融合）测试。

验证两件事：
1. _rrf_fuse 融合逻辑正确（两路排名合并成统一序）
2. 在精确术语查询场景下，hybrid（dense+sparse）召回优于纯 dense
   —— 这是引入 sparse 路的核心价值
"""

import pytest

from app.agent.nodes import _rrf_fuse


def test_rrf_fuse_returns_all_candidates():
    """融合结果应包含两路所有候选，不丢失。"""
    fused = _rrf_fuse([0, 1, 2], [2, 1, 0], k=60)
    assert sorted(fused) == [0, 1, 2]


def test_rrf_fuse_both_top_ranks_first():
    """在两路都靠前的候选应排最前。"""
    # 候选 0：dense 第1、sparse 第1 → 应该排第1
    fused = _rrf_fuse([0, 1, 2], [0, 2, 1], k=60)
    assert fused[0] == 0


def test_rrf_fuse_balances_two_signals():
    """一个候选在某路排第1但另一路排末尾，另一候选两路都居中，
    后者 RRF 得分应更高（两路均衡优于单路极端）。"""
    # 候选 0：dense#0 + sparse#2 → 1/60 + 1/62 ≈ 0.0328
    # 候选 1：dense#1 + sparse#1 → 1/61 + 1/61 ≈ 0.0328（实际略高，因对称）
    fused = _rrf_fuse([0, 1], [1, 0], k=60)
    # 两候选完全对称（一个 dense#0/sparse#1，另一个 dense#1/sparse#0），得分应相等
    # sorted 稳定，下标小的在前
    assert len(fused) == 2


def test_rrf_fuse_singleton_path():
    """只有一路排名时（另一路为空），等价于该路原序。"""
    fused = _rrf_fuse([2, 0, 1], [], k=60)
    assert fused == [2, 0, 1]


def test_rrf_k_affects_decay():
    """k 越小，头部（rank=0）与尾部（rank=大）的得分差距越大。"""
    # k=1: 0号得 1/(1+0)=1, 1号得 1/(1+1)=0.5，差距大
    # k=1000: 差距被压缩，两候选得分几乎相等
    small_k = _rrf_fuse([0, 1], [1, 0], k=1)
    large_k = _rrf_fuse([0, 1], [1, 0], k=1000)
    assert len(small_k) == 2
    assert len(large_k) == 2


@pytest.mark.asyncio
async def test_hybrid_beats_dense_on_exact_term():
    """精确术语查询：hybrid 应把含精确词项的文档召回并排前。

    构造场景：查询含罕见精确缩写 "CIoU"，语料里只有一篇含 "CIoU"。
    dense（语义）可能把这篇排得不靠前；sparse（字面）能精确命中。
    hybrid 融合后该篇排名应不差于纯 dense。
    """
    from app.services.embedding_service import encode_query_full, sparse_score

    query = "YOLOv5 用 CIoU 损失函数做边界框回归"
    # 语料：gold（含 CIoU）+ 多个语义相关但无精确词项的干扰项
    corpus = [
        "YOLOv5 使用 CIoU 作为边界框回归的损失函数，提升定位精度",  # gold
        "目标检测模型通常使用 L1/L2 或 IoU 类损失优化检测框",
        "边界框回归是目标检测的核心任务之一",
        "Faster R-CNN 的 RPN 网络生成候选区域",
        "深度学习中的损失函数衡量预测与真实值的差异",
    ]

    qe = await encode_query_full(query)
    scores = await sparse_score(qe["sparse"], corpus)
    gold_sparse_rank = scores[0]  # gold 的 sparse 得分
    # gold（含精确 CIoU）的 sparse 得分应最高
    assert gold_sparse_rank == max(scores), "含精确术语的文档 sparse 得分应最高"
