from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.state import AgentState
from app.config import get_settings
from app.services.document_service import get_user_collection
from app.services.embedding_service import encode_single
from app.services.rerank_service import rerank

settings = get_settings()

_llm: ChatOpenAI | None = None
_llm_stream: ChatOpenAI | None = None


def get_llm(streaming: bool = False) -> ChatOpenAI:
    global _llm, _llm_stream
    if streaming:
        if _llm_stream is None:
            _llm_stream = ChatOpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                streaming=True,
                max_tokens=1024,
            )
        return _llm_stream
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_tokens=512,
        )
    return _llm


async def intent_router(state: AgentState) -> AgentState:
    question = state["question"]
    user_id = state["user_id"]

    collection = get_user_collection(user_id)
    try:
        doc_count = collection.count()
    except Exception:
        doc_count = 0

    if doc_count == 0:
        state["should_retrieve"] = False
        return state

    try:
        llm = get_llm()
        messages = [
            SystemMessage(
                content=(
                    "判断用户问题是否需要从用户上传的私有文档中检索信息。"
                    "只输出 yes 或 no。问题涉及用户文档/上传资料/具体事实数据时输出 yes；"
                    "闲聊/通用知识/写代码/创作类输出 no。"
                )
            ),
            HumanMessage(content=f"问题：{question}"),
        ]
        resp = await llm.ainvoke(messages)
        state["should_retrieve"] = "yes" in resp.content.strip().lower()
    except Exception:
        state["should_retrieve"] = True
    return state


async def retrieve_documents(state: AgentState) -> AgentState:
    question = state["question"]
    user_id = state["user_id"]

    collection = get_user_collection(user_id)
    query_vec = await encode_single(question)

    try:
        results = collection.query(query_embeddings=[query_vec], n_results=settings.retrieve_top_k)
    except Exception:
        state["retrieved_docs"] = []
        state["sources"] = []
        return state

    candidates = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not candidates:
        state["retrieved_docs"] = []
        state["sources"] = []
        return state

    top_pairs = await rerank(question, candidates, top_k=settings.rerank_top_k)

    retrieved_docs: list[str] = []
    sources = []
    for orig_idx, score in top_pairs:
        retrieved_docs.append(candidates[orig_idx])
        meta = metadatas[orig_idx]
        sources.append(
            {
                "document_id": meta.get("document_id"),
                "filename": meta.get("filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "content": candidates[orig_idx][:200],
                "score": round(float(score), 4),
            }
        )

    state["retrieved_docs"] = retrieved_docs
    state["sources"] = sources
    return state


def _build_rag_prompt(question: str, context_docs: list[str]) -> list:
    context = "\n\n---\n\n".join(context_docs) if context_docs else "(无相关文档)"
    system = (
        "你是文档问答助手。基于以下文档内容回答用户问题。"
        "要求：1) 答案必须仅基于文档内容；2) 若文档无法回答请直接说明'根据当前文档无法回答'，不要编造；"
        "3) 简洁专业，中文回答。"
    )
    user = f"文档内容：\n{context}\n\n用户问题：{question}"
    return [SystemMessage(content=system), HumanMessage(content=user)]


async def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        state["answer"] = "根据当前文档无法回答该问题。"
        state["answer_tokens"] = [state["answer"]]
        return state

    llm = get_llm(streaming=True)
    messages = _build_rag_prompt(question, docs)

    tokens: list[str] = []
    async for chunk in llm.astream(messages):
        content = chunk.content
        if isinstance(content, str) and content:
            tokens.append(content)

    state["answer_tokens"] = tokens
    state["answer"] = "".join(tokens)
    return state


async def general_answer(state: AgentState) -> AgentState:
    question = state["question"]
    llm = get_llm(streaming=True)

    tokens: list[str] = []
    async for chunk in llm.astream([HumanMessage(content=question)]):
        content = chunk.content
        if isinstance(content, str) and content:
            tokens.append(content)

    state["answer_tokens"] = tokens
    state["answer"] = "".join(tokens)
    return state
