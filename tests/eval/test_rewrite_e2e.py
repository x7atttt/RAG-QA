"""Query 改写端到端实测脚本。

用 doc_user_2（简历文档）构造真实多轮指代对话，跑完整 graph，
对比"绕过改写（单轮）vs 走改写（多轮）"的检索结果，验证改写是否生效。

用法：
    .venv/Scripts/python.exe tests/eval/test_rewrite_e2e.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.agent.nodes import rewrite_query
from app.services.embedding_service import encode_single
from app.services.document_service import get_user_collection
from app.services.rerank_service import rerank

USER_ID = 2  # doc_user_2（简历文档，76 chunks）

# 构造的指代对话：轮1完整问题，轮2用指代词"它"
DIALOGUE = [
    {
        "role": "user",
        "content": "AI驱动的数据处理平台的项目背景是什么？",
    },
    {
        "role": "assistant",
        "content": "针对非技术人员面对百万行级 CSV/Excel 数据时不会写 SQL、分析效率低的痛点，"
        "开发了一款 AI 驱动的数据处理平台，用户用自然语言提问，AI 自动生成 SQL 查询。",
    },
]
# 轮2：指代"它"指代数据处理平台
FOLLOWUP = "它的技术栈有哪些？"


async def main():
    print("=" * 60)
    print("Query 改写端到端实测")
    print("=" * 60)
    print(f"轮1(完整): {DIALOGUE[0]['content']}")
    print(f"轮2(指代): {FOLLOWUP}")
    print()

    col = get_user_collection(USER_ID)
    print(f"文档库: doc_user_{USER_ID} ({col.count()} chunks)\n")

    # --- 方式A：绕过改写，直接用原始指代问题检索 ---
    print("[A] 绕过改写：直接用 '它的技术栈有哪些？' 检索")
    vec_a = await encode_single(FOLLOWUP)
    res_a = col.query(query_embeddings=[vec_a], n_results=5)
    docs_a = res_a["documents"][0]
    pairs_a = await rerank(FOLLOWUP, docs_a, top_k=3)
    print("  Top-3 检索结果:")
    for rank, (idx, score) in enumerate(pairs_a, 1):
        snippet = docs_a[idx][:80].replace("\n", " ")
        print(f"    {rank}. [{score:.4f}] {snippet}...")
    print()

    # --- 方式B：走改写节点，用改写后的 query 检索 ---
    print("[B] 走改写：rewrite_query 节点消解指代后检索")
    state = {"question": FOLLOWUP, "history": DIALOGUE}
    state = await rewrite_query(state)
    rewritten = state["rewritten_query"]
    print(f"  改写结果: {rewritten}")

    vec_b = await encode_single(rewritten)
    res_b = col.query(query_embeddings=[vec_b], n_results=5)
    docs_b = res_b["documents"][0]
    pairs_b = await rerank(rewritten, docs_b, top_k=3)
    print("  Top-3 检索结果:")
    for rank, (idx, score) in enumerate(pairs_b, 1):
        snippet = docs_b[idx][:80].replace("\n", " ")
        print(f"    {rank}. [{score:.4f}] {snippet}...")
    print()

    # --- 对比 ---
    print("=" * 60)
    print("对比")
    print("=" * 60)
    tech_keywords = ["技术栈", "Django", "FastAPI", "RBAC", "SQL 安全"]
    a_hit = any(any(kw in d for kw in tech_keywords) for d in [docs_a[i] for i, _ in pairs_a])
    b_hit = any(any(kw in d for kw in tech_keywords) for d in [docs_b[i] for i, _ in pairs_b])
    print(f"绕过改写: Top-3 含技术栈内容 = {'是' if a_hit else '否'}")
    print(f"走改写:   Top-3 含技术栈内容 = {'是' if b_hit else '否'}")
    # rerank 分数对比（改写后语义更精准，分数应更高）
    a_top1 = pairs_a[0][1] if pairs_a else 0
    b_top1 = pairs_b[0][1] if pairs_b else 0
    print(f"\nTop-1 rerank 分数: 不改写={a_top1:.4f} → 改写={b_top1:.4f}")

    if b_top1 > a_top1 * 1.5:
        print(f"\n✅ 改写生效：rerank 分数提升 {((b_top1/a_top1)-1)*100:.0f}%（语义匹配更精准）")
    elif b_hit and not a_hit:
        print("\n✅ 改写生效：指代消解后正确检索到目标内容")
    elif b_hit and a_hit:
        print("\n⚠️  改写前后都命中（dense 语义够强，指代未造成检索失败）")
    else:
        print("\n❓ 改写后仍未命中（检查改写质量或文档内容）")


if __name__ == "__main__":
    asyncio.run(main())
