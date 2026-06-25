"""InduOCRBench 检索召回评测脚本。

评测目标：验证 hybrid 检索（dense+sparse RRF）相比纯 dense 的召回提升。
数据集：qihoo360/InduOCRBench（中文企业技术文档，含表格/多栏/水印等 11 类挑战场景）

评测口径：
- 检索召回率（Hit@k / MRR），不评测生成质量（那是 RAGAS 的活，下一步）
- 用 doc_md（Hybrid Markdown）直接灌库，不走 PDF 解析，排除 OCR 误差，
  纯测检索链路（embedding 召回 + rerank 精排）
- 命中判定：检索回来的 chunk 是否包含 evidence（答案原文依据）所在文本

选题策略：
- 只选"检索友好"题：Basic Recognition / Structural Alignment / Cross-Field
  Continuity / Complex Reasoning
- 剔除对抗性/统计类题（*Attack / Counting / Aggregation），这类需 LLM 推理，
  不是检索能解决的，混入会扭曲检索指标

用法：
    python tests/eval/run_eval.py [--sample 50] [--doc-limit 30]
"""

import argparse
import asyncio
import json
import os
import random
import sys

# 确保能 import app 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import chromadb

from app.agent.nodes import _rrf_fuse
from app.config import get_settings
from app.services.embedding_service import (
    encode_query_full,
    encode_single,
    sparse_score,
)
from app.services.rerank_service import rerank

settings = get_settings()

BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "InduOCRBench", "RAG_eval")
QA_PATH = os.path.join(BENCH_DIR, "QA_pairs.jsonl")
DOC_MD_DIR = os.path.join(BENCH_DIR, "doc_md")
EVAL_CHROMA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_eval")
)
COLLECTION_NAME = "induocrbench_eval"

# 检索友好题型（保留）：靠检索能命中，不需要跨行统计/结构对抗推理
FRIENDLY_CATEGORIES = {
    "Basic Recognition",
    "Structural Alignment",
    "Cross-Field Continuity",
    "Complex Reasoning",
}


def load_friendly_qa(sample_n: int | None = None) -> list[dict]:
    """加载并筛选检索友好题。"""
    items = [json.loads(l) for l in open(QA_PATH, encoding="utf-8")]
    friendly = [it for it in items if it.get("question_category") in FRIENDLY_CATEGORIES]
    if sample_n and len(friendly) > sample_n:
        random.seed(42)  # 可复现
        friendly = random.sample(friendly, sample_n)
    return friendly


async def build_collection(qa_items: list[dict], doc_limit: int | None = None) -> chromadb.Collection:
    """把涉及的 doc_md 灌入独立 collection。

    灌库方式：按 doc_md 整文件读取 → 用项目自带 split_text 分块（与生产一致，
    含 </table> 表格保护）→ embedding 入库。这样评测的就是真实生产分块+检索链路。
    """
    from app.services.text_splitter import split_text
    from app.services.embedding_service import encode_texts

    # 清理旧评测库
    if os.path.exists(EVAL_CHROMA_DIR):
        import shutil

        shutil.rmtree(EVAL_CHROMA_DIR, ignore_errors=True)

    client = chromadb.PersistentClient(path=EVAL_CHROMA_DIR)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    # 收集涉及的文档（去重）
    filenames = list({it["filename"] for it in qa_items})
    if doc_limit:
        filenames = filenames[:doc_limit]

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict] = []
    for fn in filenames:
        fp = os.path.join(DOC_MD_DIR, fn)
        if not os.path.exists(fp):
            continue
        text = open(fp, encoding="utf-8").read().strip()
        if not text:
            continue
        chunks = split_text(
            text, strategy="recursive", chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_ids.append(f"{fn}__{i}")
            all_metas.append({"filename": fn, "chunk_index": i})

    print(f"  灌库: {len(filenames)} 文档 → {len(all_chunks)} chunks")
    # 分批 encode（避免一次性太大批 OOM）
    batch = 32
    all_vecs = []
    for s in range(0, len(all_chunks), batch):
        all_vecs.extend(await encode_texts(all_chunks[s : s + batch]))

    col.add(ids=all_ids, documents=all_chunks, embeddings=all_vecs, metadatas=all_metas)
    return col


def is_hit(retrieved_chunks: list[str], evidence: str | list, question: str) -> bool:
    """命中判定：检索 chunk 是否包含 evidence 文本。

    evidence 可能是 str 或 list[str]。表格类 evidence 是 HTML 片段，
    取其中最长的一行做子串匹配（去空白后），降低噪声。
    """
    if isinstance(evidence, list):
        ev_texts = [str(e) for e in evidence if e]
    else:
        ev_texts = [str(evidence)]
    if not ev_texts:
        return False
    # 取最长的 evidence 片段（通常是最有区分度的）
    ev = max(ev_texts, key=len)
    ev_compact = "".join(ev.split())  # 去所有空白
    if len(ev_compact) < 8:  # 太短无区分度，退而用 question 关键词
        ev_compact = "".join(question.split())[:20]
    for chunk in retrieved_chunks:
        chunk_compact = "".join(chunk.split())
        if ev_compact[:30] in chunk_compact:  # 取前30字符做指纹，避免表格换行差异
            return True
    return False


async def retrieve_dense(col, question: str, top_k: int = 3) -> list[str]:
    """纯 dense 检索（旧逻辑）：HNSW 召回 → rerank。"""
    vec = await encode_single(question)
    res = col.query(query_embeddings=[vec], n_results=settings.retrieve_top_k)
    docs = res["documents"][0]
    pairs = await rerank(question, docs, top_k=top_k)
    return [docs[i] for i, _ in pairs]


async def retrieve_dense_norerank(col, question: str, top_k: int = 3) -> list[str]:
    """纯 dense 召回，不走 rerank（直接取 HNSW 前 top_k）。

    用于对照：隔离 sparse 的价值。rerank 会掩盖召回顺序的差异，
    去掉 rerank 才能看到 dense 召回 vs hybrid 召回的真实排序质量。
    """
    vec = await encode_single(question)
    res = col.query(query_embeddings=[vec], n_results=top_k)
    return res["documents"][0]


async def retrieve_hybrid(col, question: str, top_k: int = 3) -> list[str]:
    """hybrid 检索（新逻辑）：dense 粗筛 → sparse 重排 → RRF → rerank。

    评测优化：sparse 重排在 dense Top-N 候选上做。N 取 retrieve_top_k（20）
    而非 dense_recall_top_k（50），减少 sparse encode 量（评测场景 20 已足够区分）。
    """
    qe = await encode_query_full(question)
    res = col.query(query_embeddings=[qe["dense"]], n_results=settings.retrieve_top_k)
    docs = res["documents"][0]
    if not docs:
        return []
    sp = await sparse_score(qe["sparse"], docs)
    dense_rank = list(range(len(docs)))
    sparse_rank = sorted(range(len(docs)), key=lambda i: sp[i], reverse=True)
    fused = _rrf_fuse(dense_rank, sparse_rank, settings.rrf_k)
    fused_docs = [docs[i] for i in fused]
    pairs = await rerank(question, fused_docs, top_k=top_k)
    return [fused_docs[i] for i, _ in pairs]


async def retrieve_hybrid_norerank(col, question: str, top_k: int = 3) -> list[str]:
    """hybrid 检索，不走 rerank（dense+sparse RRF 后直接取前 top_k）。

    对照组：与 dense_norerank 对比，纯看 sparse+RRF 对召回排序的改善。
    这是 sparse 路的真实价值所在（不受 reranker 掩盖）。
    """
    qe = await encode_query_full(question)
    res = col.query(query_embeddings=[qe["dense"]], n_results=settings.retrieve_top_k)
    docs = res["documents"][0]
    if not docs:
        return []
    sp = await sparse_score(qe["sparse"], docs)
    dense_rank = list(range(len(docs)))
    sparse_rank = sorted(range(len(docs)), key=lambda i: sp[i], reverse=True)
    fused = _rrf_fuse(dense_rank, sparse_rank, settings.rrf_k)
    return [docs[i] for i in fused[:top_k]]


async def evaluate(col, qa_items: list[dict], retrieve) -> dict:
    """跑一轮评测，返回 Hit@1/3/5 和 MRR。

    优化：每题只检索一次 Top5（rerank），在结果里分别判断 Hit@1/3/5，
    避免对同一题重复 rerank 三次。
    """
    ks = [1, 3, 5]
    hits = {k: 0 for k in ks}
    mrr = 0.0
    n = len(qa_items)
    name = getattr(retrieve, "__name__", "?")
    for idx, item in enumerate(qa_items):
        top5 = await retrieve(col, item["question"], top_k=5)
        # MRR：找命中位置
        rank_found = 0
        for r, ch in enumerate(top5, 1):
            if is_hit([ch], item["evidence"], item["question"]):
                rank_found = r
                break
        if rank_found:
            mrr += 1.0 / rank_found
        # Hit@k：前 k 个里是否命中
        for k in ks:
            if is_hit(top5[:k], item["evidence"], item["question"]):
                hits[k] += 1
        if (idx + 1) % 10 == 0:
            print(f"    {name} progress: {idx+1}/{n}")
    return {
        "n": n,
        "Hit@1": round(hits[1] / n, 4),
        "Hit@3": round(hits[3] / n, 4),
        "Hit@5": round(hits[5] / n, 4),
        "MRR": round(mrr / n, 4),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50, help="抽样题数")
    parser.add_argument("--doc-limit", type=int, default=None, help="限制灌库文档数")
    args = parser.parse_args()

    print("=" * 60)
    print("InduOCRBench 检索召回评测")
    print("=" * 60)

    qa = load_friendly_qa(args.sample)
    print(f"筛选检索友好题: {len(qa)} 题 (从 2071 中筛)")

    print("\n[1/5] 灌库...")
    col = await build_collection(qa, args.doc_limit)
    print(f"  collection: {col.count()} chunks")

    # 4 组对照：rerank 有/无 × dense/hybrid
    methods = {
        "dense_norerank": retrieve_dense_norerank,
        "hybrid_norerank": retrieve_hybrid_norerank,
        "dense_rerank": retrieve_dense,
        "hybrid_rerank": retrieve_hybrid,
    }
    results = {}
    step = 2
    for name, fn in methods.items():
        print(f"\n[{step}/5] 评测 {name}...")
        results[name] = await evaluate(col, qa, fn)
        print(f"  结果: {results[name]}")
        step += 1

    print("\n" + "=" * 60)
    print("对比汇总")
    print("=" * 60)
    print(f"{'方法':<20} | {'Hit@1':>7} | {'Hit@3':>7} | {'Hit@5':>7} | {'MRR':>7}")
    print("-" * 60)
    for name in methods:
        r = results[name]
        print(f"{name:<20} | {r['Hit@1']:>7} | {r['Hit@3']:>7} | {r['Hit@5']:>7} | {r['MRR']:>7}")

    # 关键对比：无 rerank 时 hybrid vs dense（隔离 sparse 价值）
    print("\n--- 关键对比：无 rerank（隔离 sparse 价值）---")
    for k in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:
        d = results["dense_norerank"][k]
        h = results["hybrid_norerank"][k]
        delta = round((h - d) * 100, 2)
        sign = "+" if delta >= 0 else ""
        print(f"  {k}: dense={d} → hybrid={h} ({sign}{delta}%)")

    result = {
        "sample_size": len(qa),
        "chunks": col.count(),
        "results": results,
    }
    out = os.path.join(os.path.dirname(__file__), "eval_result.json")
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    asyncio.run(main())
