import asyncio
import threading

from FlagEmbedding import FlagReranker

from app.config import get_settings

settings = get_settings()

_reranker: FlagReranker | None = None
_lock = threading.Lock()


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = FlagReranker(settings.rerank_model_path, use_fp16=False)
    return _reranker


def _rerank_sync(query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, float]]:
    if not documents:
        return []
    model = get_reranker()
    pairs = [[query, doc] for doc in documents]
    scores = model.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k]


async def rerank(query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, float]]:
    return await asyncio.to_thread(_rerank_sync, query, documents, top_k)
