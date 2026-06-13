import asyncio
import threading
from typing import Any

from FlagEmbedding import BGEM3FlagModel

from app.config import get_settings

settings = get_settings()

_model: BGEM3FlagModel | None = None
_lock = threading.Lock()


def get_embedding_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = BGEM3FlagModel(settings.embedding_model_path, use_fp16=False)
    return _model


def _encode_sync(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    output: dict[str, Any] = model.encode(
        texts, batch_size=batch_size, max_length=8192, return_dense=True
    )
    return output["dense_vecs"].tolist()


async def encode_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    return await asyncio.to_thread(_encode_sync, texts, batch_size)


async def encode_single(text: str) -> list[float]:
    result = await encode_texts([text])
    return result[0]
