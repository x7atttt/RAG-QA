"""BGE-M3 → LangChain Embeddings 适配器。

RAGAS 的 Answer Relevancy 指标需要算文本语义相似度（对比"从答案反生成的问题"
与原问题的相似度），需要一个 embedding。项目用本地 BGE-M3（GPU 加速），
这里包一层 langchain_core.embeddings.Embeddings 接口，让 RAGAS 能复用。

不引入额外的 embedding 服务（如 OpenAI text-embedding-3）——复用项目的 BGE-M3，
保证评测与生产用同一套 embedding，口径一致。
"""

import asyncio

from langchain_core.embeddings import Embeddings

from app.services.embedding_service import encode_texts


class BGEM3Embeddings(Embeddings):
    """把 BGE-M3 包成 LangChain Embeddings 接口。

    RAGAS 通过 LangchainEmbeddingsWrapper 调用本类的 embed_documents/embed_query。
    底层复用 app.services.embedding_service.encode_texts（单例模型 + GPU）。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """同步批量 embed（RAGAS 内部可能在同步上下文调用）。"""
        # encode_texts 是 async，这里用 asyncio.run 适配同步接口
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在事件循环中（如 async 评测脚本），用临时新循环跑
                return asyncio.run(encode_texts(texts))
        except RuntimeError:
            pass
        return asyncio.run(encode_texts(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await encode_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        result = await encode_texts([text])
        return result[0]
