from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    general_answer,
    generate_answer,
    grade_documents,
    intent_router,
    retrieve_documents,
    rewrite_query,
    transform_query,
)
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intent_router", intent_router)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("general_answer", general_answer)

    graph.set_entry_point("intent_router")
    # 需检索时：先改写 query（多轮指代消解）再检索
    graph.add_conditional_edges(
        "intent_router",
        lambda state: state.get("should_retrieve", False),
        {True: "rewrite_query", False: "general_answer"},
    )
    graph.add_edge("rewrite_query", "retrieve_documents")
    # CRAG 循环：检索 → 评分 → (低相关且未达上限) → 改写变体 → 回检索（重查）
    #                   → (高相关 或 达上限) → 生成
    graph.add_edge("retrieve_documents", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("meta", {}).get("should_retry", False),
        {True: "transform_query", False: "generate_answer"},
    )
    graph.add_edge("transform_query", "retrieve_documents")  # 回边（循环）
    graph.add_edge("generate_answer", END)
    graph.add_edge("general_answer", END)
    return graph.compile()
