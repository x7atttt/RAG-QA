import asyncio
import logging
import threading
import warnings
from typing import Any

from FlagEmbedding import BGEM3FlagModel

from app.config import get_settings

settings = get_settings()

# 过滤 FlagEmbedding 内部的 XLMRobertaTokenizerFast 提示（库内部用 encode+pad
# 而非 __call__，属于库的实现细节，不影响功能）
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

_model: BGEM3FlagModel | None = None
_lock = threading.Lock()


def _select_device() -> str:
    """选择推理设备：有 CUDA 用 cuda，否则 cpu。
    FlagEmbedding 的 BGEM3FlagModel 接受 device 字符串（cuda / cuda:0 / cpu）。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_embedding_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                device = _select_device()
                # use_fp16 仅在 GPU 上生效（CPU 上 fp16 反而更慢且精度差），故按设备自适应
                use_fp16 = device.startswith("cuda")
                _model = BGEM3FlagModel(
                    settings.embedding_model_path,
                    use_fp16=use_fp16,
                    device=device,
                )
                logging.getLogger("docqa.embedding").info(
                    f"BGE-M3 加载完成 (device={device}, fp16={use_fp16})"
                )
    return _model


def _encode_sync(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    # max_length 与 chunk_size 对齐：chunk 默认 500 字符，512 token 足够覆盖。
    # 之前用 8192 导致 BGE-M3 在 CPU 上每个 batch 推理极慢（17 chunks 21s），
    # 调到 512 后 CPU 提速约 5-7 倍，GPU 下更是 <1s。
    output: dict[str, Any] = model.encode(
        texts, batch_size=batch_size, max_length=settings.embedding_max_length, return_dense=True
    )
    return output["dense_vecs"].tolist()


async def encode_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    return await asyncio.to_thread(_encode_sync, texts, batch_size)


async def encode_single(text: str) -> list[float]:
    result = await encode_texts([text])
    return result[0]


# ---------------------------------------------------------------------------
# 稀疏检索（sparse / lexical_weights）
# ---------------------------------------------------------------------------
# BGE-M3 同时支持 dense / sparse(lexical_weights) / colbert 三种向量表示。
# 之前只用了 dense（return_dense=True），稀疏能力闲置。
# 这里启用 sparse：lexical_weights 是 {token_id: weight} 字典，本质是
# BGE-M3 学习出的稀疏词项权重，做 query-doc 点积即 BM25 风格的相关性打分。
# 与 Chroma 自带的纯词频 BM25 不同，这是模型 learned 的稀疏表示，质量更高。
# ---------------------------------------------------------------------------


def _encode_full_sync(
    texts: list[str], batch_size: int = 16
) -> tuple[list[list[float]], list[dict[int, float]]]:
    """同时返回 dense 向量与 sparse lexical_weights。

    用于检索阶段：query 需要同时拿到两路表示做 RRF 融合。
    lexical_weights 的 key 是 int token_id（未 decode 成 token 字符串），
    保持 int 便于跨 query/doc 做点积匹配（decode 后字符串可能有微小差异）。
    """
    if not texts:
        return [], []
    model = get_embedding_model()
    output: dict[str, Any] = model.encode(
        texts,
        batch_size=batch_size,
        max_length=settings.embedding_max_length,
        return_dense=True,
        return_sparse=True,
    )
    dense = output["dense_vecs"].tolist()
    sparse = output["lexical_weights"]
    # 单条输入时 FlagEmbedding 会把 lexical_weights 压平为 dict 而非 [dict]，统一成 list
    if isinstance(sparse, dict):
        sparse = [sparse]
    # key 统一成 int（FlagEmbedding 返回的 key 可能是 int 或 numpy 标量）
    sparse = [{int(k): float(v) for k, v in lw.items()} for lw in sparse]
    return dense, sparse


async def encode_query_full(text: str) -> dict[str, Any]:
    """查询编码：返回 dense 向量 + sparse lexical_weights，供 hybrid 检索。"""
    dense_list, sparse_list = await asyncio.to_thread(_encode_full_sync, [text])
    return {"dense": dense_list[0], "sparse": sparse_list[0]}


def _sparse_dot(lw1: dict[int, float], lw2: dict[int, float]) -> float:
    """两个 lexical_weights 的点积得分（BM25 风格相关性）。

    只遍历较短的一方，O(min(len))。token_id 命中即累乘权重。
    """
    if len(lw1) > len(lw2):
        lw1, lw2 = lw2, lw1
    return sum(w * lw2.get(tid, 0.0) for tid, w in lw1.items())


def sparse_score_sync(
    query_sparse: dict[int, float], doc_texts: list[str]
) -> list[float]:
    """对一批候选文本计算与 query 的稀疏匹配得分。

    复用同一个 BGE-M3 实例 encode 出每个 doc 的 lexical_weights，
    再与 query_sparse 点积。供 retrieve_documents 的 sparse 路（候选集内重排）使用。
    """
    if not doc_texts:
        return []
    model = get_embedding_model()
    output: dict[str, Any] = model.encode(
        doc_texts,
        batch_size=16,
        max_length=settings.embedding_max_length,
        return_dense=False,
        return_sparse=True,
    )
    doc_sparse = output["lexical_weights"]
    if isinstance(doc_sparse, dict):
        doc_sparse = [doc_sparse]
    return [
        _sparse_dot(query_sparse, {int(k): float(v) for k, v in lw.items()})
        for lw in doc_sparse
    ]


async def sparse_score(
    query_sparse: dict[int, float], doc_texts: list[str]
) -> list[float]:
    return await asyncio.to_thread(sparse_score_sync, query_sparse, doc_texts)
