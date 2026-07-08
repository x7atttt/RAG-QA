from typing import Any, TypedDict


class SourceItem(TypedDict):
    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float


class HistoryItem(TypedDict):
    """单条历史对话，role ∈ {'user', 'assistant'}。"""
    role: str
    content: str


class AgentState(TypedDict, total=False):
    user_id: int
    question: str
    # query 改写：多轮指代消解后的检索 query（仅检索用，生成仍用原始 question）
    rewritten_query: str
    # CRAG 变体：基于"检索结果差"改写的同义/换角度 query（仅重查触发，优先级高于 rewritten_query）
    transformed_query: str
    # CRAG 循环计数：低相关重查的次数，达 crag_max_attempts 后不再重查走 fallback
    retrieval_attempts: int
    # 意图路由
    should_retrieve: bool
    # 工具路由（spec/20260706-tool-calling.md）：tool_router 决策是否调文档元信息工具
    # None=不调工具走检索/对话；"doc_meta"=调文档元信息工具
    tool_call: str | None
    # 文档元信息工具的查询意图（tool_router 提取）：list / recent
    doc_meta_intent: str
    # 文档元信息工具返回结果（format_for_prompt 后的文本）
    doc_meta_result: str
    # 联网搜索工具返回结果（format_for_prompt 后的文本）；非空时进 _build_web_search_prompt
    web_search_result: str
    # 检索
    retrieved_docs: list[str]
    # 候选池（rerank 前的 top-20），供 fallback 路径使用更多上下文
    fallback_docs: list[str]
    sources: list[SourceItem]
    # 多轮上下文：最近 N 轮历史（正序，最旧在前）
    history: list[HistoryItem]
    # 会话摘要：长对话老上下文的压缩（达阈值异步生成），注入 system prompt 做长期记忆
    summary: str
    # 是否开启 DeepSeek thinking 模式（用户自定义）
    thinking: bool
    # 答案
    answer_tokens: list[str]
    answer: str
    # 推理内容（DeepSeek reasoner / thinking 模式）
    reasoning_tokens: list[str]
    reasoning: str
    error: str | None
    meta: dict[str, Any]
