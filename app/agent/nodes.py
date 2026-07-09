import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.memory import truncate_by_token_budget
from app.agent.state import AgentState
from app.config import get_settings
from app.services.document_service import get_user_collection
from app.services.embedding_service import encode_query_full, encode_single, sparse_score
from app.services.llm_provider import astream_chat, chat
from app.services.rerank_service import rerank

settings = get_settings()


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
        messages = [
            SystemMessage(
                content=(
                    "判断用户问题是否需要从用户上传的私有文档中检索信息。只输出 yes 或 no。\n"
                    "输出 yes 的情况：问题询问文档内容、要求总结/查找/对比具体信息、提到'文档/资料/文件/上面提到'等；"
                    "包含指代词（它/这个/那个）且上下文可能指向文档内容时也输出 yes。\n"
                    "输出 no 的情况：纯闲聊、写代码、通用百科知识、创作类请求。\n"
                    "拿不准时倾向输出 yes（宁可多检索）。"
                )
            ),
            HumanMessage(content=f"问题：{question}"),
        ]
        resp = await chat(messages)
        state["should_retrieve"] = "yes" in resp.strip().lower()
    except Exception:
        state["should_retrieve"] = True
    return state


async def tool_router(state: AgentState) -> AgentState:
    """工具路由（spec/20260706-tool-calling.md 决策 3）：判断是否调文档元信息工具。

    定位：串在 intent_router 之前（显式命令优先）。tool_router 先判断是否是
    联网搜索命令或文档元信息问题，都不是才走 intent_router 判断要不要检索。
    两者职责分离，每节点一件事。

    元层问题示例：
      - "我上传过哪些文档/PDF/Markdown"
      - "最近一周加了什么文件"
      - "哪份文档最长/最大"
    这类问题的答案不在 chunk 里（检索失效），而在文档元数据里。

    输出（写入 state）：
      tool_call = "doc_meta"：调文档元信息工具，跳过检索链路
      tool_call = None：不调工具，走后续检索/对话
      doc_meta_intent = "list" | "recent"：工具查询意图
    """
    question = state["question"]

    # 显式联网搜索命令优先：用户已经明确要求联网查询时，不等 CRAG 失败再触发
    if _is_explicit_web_search_command(question):
        state["tool_call"] = "web_search"
        return state

    try:
        messages = [
            SystemMessage(
                content=(
                    "判断用户问题是否在询问【文档元信息】（关于文档本身的事实，而非文档内容）。\n"
                    "判断为是的示例：\n"
                    "- 我上传过哪些文档/PDF/Markdown/Word\n"
                    "- 最近一周/今天加了什么文件\n"
                    "- 哪份文档最长/最大/最旧\n"
                    "- 一共有多少份文档\n"
                    "判断为否的示例（这些是内容问题，走检索）：\n"
                    "- 文档里讲了什么 / 总结一下 / 第三条是什么\n"
                    "- 加密方案是什么 / 有没有提到 X\n"
                    "只输出一行 JSON：\n"
                    '  调工具：{"call": true, "intent": "list"}（或 "intent": "recent"，'
                    '"recent" 用于含"最近/今天/本周"等时间限定的问题）\n'
                    '  不调工具：{"call": false}'
                )
            ),
            HumanMessage(content=f"问题：{question}"),
        ]
        resp = await chat(messages, max_tokens=50)

        # 容错：LLM 可能输出多余文本，取第一个 JSON 对象
        resp = resp.strip()
        start, end = resp.find("{"), resp.rfind("}")
        if start == -1 or end == -1:
            decision = {"call": False}
        else:
            decision = json.loads(resp[start : end + 1])

        if decision.get("call") is True:
            state["tool_call"] = "doc_meta"
            state["doc_meta_intent"] = decision.get("intent", "list")
        else:
            state["tool_call"] = None
    except Exception:
        # 决策失败不阻断——降级走检索链路（顶多答得不准，不会崩）
        state["tool_call"] = None
    return state


_EXPLICIT_WEB_SEARCH_RE = re.compile(
    r"(联网|上网|在线|最新|新闻|刚刚|今天|昨天|目前|现在|实时)"
    r"(搜|查|搜一下|查一下|搜索一下|查一下|查找)?",
    re.IGNORECASE,
)


def _is_explicit_web_search_command(text: str) -> bool:
    """判断用户输入是否为显式联网搜索命令。

    优先级高于 tool_router 的 LLM 决策：
    用户已经明确表达“联网搜索/在线查/查最新消息”，应直接触发 web_search，
    不必等到 CRAG 失败后才作为兜底。
    """
    return bool(_EXPLICIT_WEB_SEARCH_RE.search(text or ""))


async def rewrite_query(state: AgentState) -> AgentState:
    """多轮指代消解：结合历史把指代词改写成可独立检索的完整 query。

    场景：用户上一轮问"文档里有哪些规则"，这一轮问"第三条是什么"，
    直接用"第三条"检索必然失败。本节点把"第三条"结合历史改写成
    "文档里第3条规则的具体内容"。

    设计：
    - 仅当 history 非空才调 LLM 改写（首问无历史，省一次调用）
    - 改写结果存 rewritten_query，供 retrieve_documents 使用
    - 生成答案仍用原始 question（回答自然，用户问的是原话）
    - 容错：LLM 异常/输出异常时降级用原始 question，不阻断流程
    """
    question = state["question"]
    history = state.get("history", [])

    # 无历史 → 无指代可消解，直接用原问题
    if not history:
        state["rewritten_query"] = question
        return state

    try:
        # 把历史拼成对话格式（取改写专用轮数，够消解指代又省 token）
        recent = history[-settings.rewrite_history_rounds * 2 :]
        dialogue = "\n".join(
            f"{'用户' if h['role'] == 'user' else '助手'}: {h['content'][:200]}"
            for h in recent
            if h.get("content")
        )
        messages = [
            SystemMessage(
                content=(
                    "你是 query 改写助手。根据对话历史，把用户的最新问题改写成一个"
                    "可以独立检索的完整问题（消解指代词如 它/这个/第几条/上面提到）。\n"
                    "要求：\n"
                    "1) 只输出改写后的问题，不要任何解释；\n"
                    "2) 保持原意，不要添加无关内容；\n"
                    "3) 如果问题本身已完整无指代，原样输出。"
                )
            ),
            HumanMessage(content=f"对话历史：\n{dialogue}\n\n最新问题：{question}\n\n改写后："),
        ]
        resp = await chat(messages)
        # 去除首尾可能的多余引号/句号（LLM 有时会用引号包裹改写结果）
        # 含中文弯引号 “ ” ‘ ’
        rewritten = resp.strip().strip("\"'“”‘’。")
        state["rewritten_query"] = rewritten or question
    except Exception:
        # 改写失败不阻断检索，降级用原始问题
        state["rewritten_query"] = question
    return state


async def transform_query(state: AgentState) -> AgentState:
    """CRAG 重查：检索得分低 → 基于失败 query + 弱命中片段生成同义/换角度变体。

    与 rewrite_query 的区别：
    - rewrite_query 消解指代（"它/第三条"→完整问题），基于对话历史，无失败信号
    - transform_query 基于"检索结果差"做语义扩展（同义词/换角度），基于召回的弱命中片段，
      仅在 CRAG 重查时触发

    设计：
    - 输入失败 query + 弱命中 top-3 片段（即便 score 低，可能含相关术语可作线索）
    - 输出语义等价但表达不同的检索变体，喂给 retrieve_documents 重查
    - 容错：LLM 异常/输出异常时复用原 query，不阻断重查流程
    """
    question = state["question"]
    # 基线：优先用已有的改写结果（指代消解后的），避免丢失 rewrite 的成果
    base = state.get("transformed_query") or state.get("rewritten_query") or question
    # 弱命中片段：召回的 top-3（即便 score 低，可能含相关术语可作线索）
    weak_hits = (state.get("retrieved_docs", []) or [])[:3]
    hits_text = "\n".join(f"- {d[:120]}" for d in weak_hits) or "(无召回)"

    try:
        messages = [
            SystemMessage(
                content=(
                    "用户的检索 query 召回结果相关度低。请改写成一个语义等价但表达不同的检索 query，"
                    "尝试：同义词替换、换角度描述、补充专业术语、拆解/重组关键词。\n"
                    "要求：\n"
                    "1) 只输出一个改写后的 query，不要任何解释；\n"
                    "2) 保留原意，不要臆造文档里没有的实体；\n"
                    "3) 可参考下方弱相关片段里的术语作为线索。"
                )
            ),
            HumanMessage(content=f"原 query：{base}\n弱相关片段：\n{hits_text}\n\n改写后："),
        ]
        resp = await chat(messages)
        # 去除首尾可能的多余引号/句号（含中文弯引号）
        variant = resp.strip().strip("\"'“”‘’。")
        state["transformed_query"] = variant or base
    except Exception:
        # 改写失败不阻断，复用原 query 重查（至少换个检索角度的机会）
        state["transformed_query"] = base
    return state


def _rrf_fuse(
    dense_rank: list[int], sparse_rank: list[int], k: int
) -> list[int]:
    """Reciprocal Rank Fusion：把 dense/sparse 两路的排名融合成统一候选序。

    dense_rank[i] 表示候选 i 在 dense 路的排名（0=最相关）；
    sparse_rank 同理。RRF 得分 = Σ 1/(k + rank)，得分越高越相关。
    返回按 RRF 得分降序的候选下标列表。
    """
    scores: dict[int, float] = {}
    for ranks in (dense_rank, sparse_rank):
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


async def retrieve_documents(state: AgentState) -> AgentState:
    """Hybrid 检索：dense（Chroma HNSW）+ sparse（BGE-M3 lexical_weights）RRF 融合。

    流程（"dense 粗筛 → sparse 重排 → RRF 融合 → reranker 精排"）：
      1. BGE-M3 同时编码 query 的 dense 向量与 sparse lexical_weights
      2. dense 路：Chroma HNSW 召回 Top-N 候选（N=dense_recall_top_k=50）
      3. sparse 路：对这批候选重新算 lexical 匹配得分，给 sparse 排名
         （不遍历全库，只在 dense 候选集内重排，控制计算量）
      4. RRF 融合两路排名 → 取 Top-K（retrieve_top_k=20）
      5. BGE-Reranker 精排 → Top-3 喂生成

    相比纯 dense：sparse 路对精确术语/关键词/缩写的强匹配能补救 dense 的语义漂移，
    把"字面没对上但语义相关"和"字面正好对上"两类命中都纳入候选。
    """
    question = state["question"]
    # 检索用 query 优先级：CRAG 变体（重查触发）> 指代消解后的 query > 原始 question
    # 生成答案时仍用原始 question（回答针对用户原话）
    search_query = state.get("transformed_query") or state.get("rewritten_query") or question
    user_id = state["user_id"]

    collection = get_user_collection(user_id)

    # 1. query 双编码：dense + sparse
    query_enc = await encode_query_full(search_query)
    query_vec = query_enc["dense"]
    query_sparse = query_enc["sparse"]

    try:
        # 2. dense 路：HNSW 召回候选（扩大到 dense_recall_top_k）
        results = collection.query(query_embeddings=[query_vec], n_results=settings.dense_recall_top_k)
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

    # 3. sparse 路：对候选集算 lexical 匹配得分
    sparse_scores = await sparse_score(query_sparse, candidates)
    # dense 路的排名就是 Chroma 返回顺序（HNSW 已按相似度排好）
    dense_rank = list(range(len(candidates)))
    # sparse 路按得分降序排
    sparse_rank = sorted(range(len(candidates)), key=lambda i: sparse_scores[i], reverse=True)

    # 4. RRF 融合 → 取 Top-K（retrieve_top_k）
    fused_order = _rrf_fuse(dense_rank, sparse_rank, settings.rrf_k)
    top_idxs = fused_order[: settings.retrieve_top_k]
    fused_candidates = [candidates[i] for i in top_idxs]
    fused_metas = [metadatas[i] for i in top_idxs]

    # 5. reranker 精排（用改写后的 query，与召回保持一致）
    top_pairs = await rerank(search_query, fused_candidates, top_k=settings.rerank_top_k)

    retrieved_docs: list[str] = []
    sources = []
    for orig_idx, score in top_pairs:
        retrieved_docs.append(fused_candidates[orig_idx])
        meta = fused_metas[orig_idx]
        sources.append(
            {
                "document_id": meta.get("document_id"),
                "filename": meta.get("filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "content": fused_candidates[orig_idx][:200],
                "score": round(float(score), 4),
            }
        )

    state["retrieved_docs"] = retrieved_docs
    state["sources"] = sources
    return state


def _history_to_messages(history: list[dict]) -> list:
    """把 state['history'] 转成 LangChain 消息列表（正序：最旧在前）。

    先按 token 预算 + 轮数双约束截断（保留最新历史），再转消息。
    """
    trimmed = truncate_by_token_budget(
        history, settings.history_token_budget, settings.max_history_rounds
    )
    msgs = []
    for item in trimmed:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def _inject_summary(messages: list, summary: str | None) -> list:
    """把会话摘要作为 system 上下文插入消息列表开头（system 之后、历史之前）。

    摘要用于消解指代词、保持长对话连贯；回答仍以文档内容为准。
    仅当 summary 非空时插入。
    """
    if not summary:
        return messages
    summary_msg = SystemMessage(
        content=(
            f"[会话历史摘要]\n{summary}\n\n"
            "（以上为本会话早期对话的要点，用于理解指代与背景；"
            "回答仍以本次提供的文档内容为准。）"
        )
    )
    # 插在第一个 system 消息之后
    if messages and isinstance(messages[0], SystemMessage):
        return [messages[0], summary_msg] + messages[1:]
    return [summary_msg] + messages


def _build_rag_prompt(
    question: str,
    context_docs: list[str],
    history: list[dict] | None = None,
    summary: str | None = None,
) -> list:
    context = "\n\n---\n\n".join(context_docs) if context_docs else "(无相关文档)"
    system = (
        "你是文档问答助手。基于以下文档内容回答用户问题。\n"
        "要求：\n"
        "1) 若文档中有直接答案，请基于文档内容回答；\n"
        "2) 若文档提供了背景信息但无直接答案（如用户询问建议、评价、改进方案），"
        "请结合文档中的实际情况给出具体、有针对性的分析和建议；\n"
        "3) 回答开头用一句话说明依据来源（如『基于您上传的简历内容』）；\n"
        "4) 不要编造文档中没有的信息；简洁专业，中文回答。"
    )
    user = f"文档内容：\n{context}\n\n用户问题：{question}"
    # 消息顺序：system → [摘要] → 历史 → 当前 human（让模型理解指代与上下文）
    messages = [SystemMessage(content=system)]
    if history:
        messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=user))
    return _inject_summary(messages, summary)


def _build_fallback_prompt(
    question: str,
    context_docs: list[str],
    history: list[dict] | None = None,
    summary: str | None = None,
) -> list:
    """检索无直接命中时的降级 prompt：结合文档背景 + 常识给出有帮助的回答。

    与严格 RAG 的区别：允许模型在"文档未直接回答"时，基于文档提供的背景信息
    （如简历内容、项目描述）+ 通用知识给出建议/分析，而不是硬拒绝。
    典型场景：用户上传简历后问"怎么改进我的简历"——文档里有简历内容，
    但没有现成的改进建议，此时应结合简历实际情况给针对性建议。
    """
    context = "\n\n---\n\n".join(context_docs) if context_docs else "(用户未上传相关文档)"
    system = (
        "你是一个智能助手。用户上传了以下文档作为参考背景。\n"
        "请根据用户问题作答：\n"
        "1) 若问题能从文档直接找到答案，请基于文档内容回答；\n"
        "2) 若文档未直接涉及该问题（如询问建议、评价、改进方案），请结合文档中可见的实际情况"
        "（如简历内容、项目细节）与你的通用知识，给出具体、有针对性的回答；\n"
        "3) 回答开头用一句话说明依据来源（如『基于您上传的简历内容』或『文档未直接涉及，以下为通用建议』）；\n"
        "4) 简洁专业，中文回答。"
    )
    user = f"文档背景：\n{context}\n\n用户问题：{question}"
    messages = [SystemMessage(content=system)]
    if history:
        messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=user))
    return _inject_summary(messages, summary)


def _build_web_search_prompt(
    question: str,
    web_context: str,
    fallback_docs: list[str] | None = None,
    history: list[dict] | None = None,
    summary: str | None = None,
) -> list:
    """联网搜索结果作为主 context 的 prompt（spec/20260706 决策 5）。

    触发场景：CRAG 重查仍失败 → 知识库无相关内容 → 调 Tavily 搜网络兜底。
    与 _build_fallback_prompt 的区别：fallback 凭常识 + 残余弱命中片段作答（易幻觉），
    本 prompt 用真实的网络检索结果作为事实依据，降低幻觉。

    来源标注由 system prompt 强制：网络结果 vs 残余文档片段分开说明。
    """
    fallback_part = (
        f"\n\n参考文档（弱相关）：\n{'---'.join(fallback_docs)}"
        if fallback_docs
        else ""
    )
    system = (
        "你是一个智能助手。用户的知识库中没有直接回答该问题的内容，已通过联网搜索获取了以下资料。\n"
        "请基于联网搜索结果回答用户问题：\n"
        "1) 答案必须基于搜索结果，不要编造未在结果中出现的信息；\n"
        "2) 回答开头标注『以下内容来自网络搜索，非您上传的文档』；\n"
        "3) 引用具体来源时附上对应链接；\n"
        "4) 搜索结果不足时坦诚说明，不强行编造。"
    )
    user = f"联网搜索结果：\n{web_context}{fallback_part}\n\n用户问题：{question}"
    messages = [SystemMessage(content=system)]
    if history:
        messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=user))
    return _inject_summary(messages, summary)


def _build_doc_meta_prompt(
    question: str,
    doc_meta_context: str,
    history: list[dict] | None = None,
    summary: str | None = None,
) -> list:
    """文档元信息工具结果的 prompt。

    触发场景：tool_router 判定问题为元层问题（"我有哪些文档"）→ 调 doc_meta 工具 →
    用本 prompt 把工具返回的元信息列表喂给 LLM 生成自然语言回答。
    """
    system = (
        "你是一个智能助手。系统已根据用户问题查询了其上传文档的元信息。\n"
        "请基于以下元信息，用自然语言回答用户问题：\n"
        "1) 直接回答用户问的（数量/文件名/时间等），不要展开文档内容；\n"
        "2) 信息以列表形式呈现时保持清晰可读；\n"
        "3) 简洁专业，中文回答。"
    )
    user = f"文档元信息：\n{doc_meta_context}\n\n用户问题：{question}"
    messages = [SystemMessage(content=system)]
    if history:
        messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=user))
    return _inject_summary(messages, summary)


def grade_documents(state: AgentState) -> AgentState:
    """CRAG 评分节点（纯逻辑，不调 LLM）：检查检索 top score 决定是否重查。

    给 graph.py 的条件边用：score 低且未达重查上限 → 标记 retry（走 transform_query 重查）；
    否则 → 标记 generate（走 generate_answer 生成）。
    chat_service 的手动编排路径不经过此节点（它自己用 while 循环控制），
    这里仅保持 graph 声明一致 + 便于面试展示完整 CRAG 循环图。
    """
    sources = state.get("sources", [])
    top_score = sources[0].get("score", 0) if sources else 0
    attempts = state.get("retrieval_attempts", 0)
    # 低相关 + 有召回 + 未达重查上限 → 重查
    should_retry = (
        bool(sources)
        and top_score < settings.retrieval_score_threshold
        and attempts < settings.crag_max_attempts
    )
    if should_retry:
        state["retrieval_attempts"] = attempts + 1
    state["meta"] = {**(state.get("meta") or {}), "should_retry": should_retry}
    return state


async def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])
    sources = state.get("sources", [])
    history = state.get("history", [])
    thinking = bool(state.get("thinking", False))

    # 根据检索结果相关度选择 prompt 策略：
    # - 高相关（top score ≥ 0.5）：严格 RAG，仅基于文档回答
    # - 低相关 / 无结果：降级 fallback，结合文档背景 + 常识给有帮助的回答
    #   典型场景：用户上传简历后问"怎么改进简历"——文档有简历内容但无现成建议，
    #   严格 RAG 会硬拒绝，fallback 让模型结合简历实际情况给针对性建议。
    top_score = sources[0].get("score", 0) if sources else 0
    if docs and top_score >= settings.retrieval_score_threshold:
        messages = _build_rag_prompt(question, docs, history, summary)
    else:
        messages = _build_fallback_prompt(question, docs, history, summary)

    # 流式生成：thinking 模式下 reasoning_content 先于 content 返回（由 chat_service 捕获）
    tokens: list[str] = []
    async for event, text in astream_chat(messages, thinking=thinking):
        if event == "content":
            tokens.append(text)

    state["answer_tokens"] = tokens
    state["answer"] = "".join(tokens)
    return state


async def general_answer(state: AgentState) -> AgentState:
    question = state["question"]
    history = state.get("history", [])
    thinking = bool(state.get("thinking", False))

    messages = _history_to_messages(history)
    messages.append(HumanMessage(content=question))

    tokens: list[str] = []
    async for event, text in astream_chat(messages, thinking=thinking):
        if event == "content":
            tokens.append(text)

    state["answer_tokens"] = tokens
    state["answer"] = "".join(tokens)
    return state
