"""RAGAS 端到端评测（生成层）：Faithfulness + Answer Relevancy。

补齐"真正的端到端"——含 LLM 生成的答案质量评测。之前评测一二四都是检索层
（Hit Rate/MRR），本评测测**生成层**：
- Faithfulness：答案有没有幻觉（是否忠实于检索内容）
- Answer Relevancy：答案切不切题

口径：OCR 库（data/chroma_eval_ocr，MinerU 解析结果），与 OCR 链路 Hit@3=71% 可比。

流程（每题）：
1. retrieved_contexts = retrieve_top5(col, question)        ← 复用 dense+rerank
2. messages = _build_rag_prompt(question, retrieved_contexts) ← 生产严格 RAG prompt
3. response = await chat(messages, thinking=False)          ← 生成答案
4. 组装 SingleTurnSample(user_input, response, retrieved_contexts, reference)
批量喂 ragas.evaluate(metrics=[Faithfulness, AnswerRelevancy], llm=judge, embeddings=...)

用法：
    python tests/eval/run_eval_ragas.py [--sample 50]
"""

import argparse
import asyncio
import json
import os
import random
import sys

# vertexai stub：ragas 0.4.3 写死 import langchain_community.chat_models.vertexai，
# 但 langchain-community 0.4 已移除该路径。注入空 stub 让 import 通过（不影响功能）。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import types  # noqa: E402

_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAIStub:  # pragma: no cover
    def __init__(self, *a, **k):
        raise NotImplementedError("ChatVertexAI stub, not used")


_stub.ChatVertexAI = _ChatVertexAIStub
sys.modules["langchain_community.chat_models.vertexai"] = _stub

import chromadb  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.evaluation import evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import AnswerRelevancy, Faithfulness  # noqa: E402

from app.agent.nodes import _build_rag_prompt  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.services.embedding_service import encode_single  # noqa: E402
from app.services.llm_provider import chat  # noqa: E402
from app.services.ragas_embed_adapter import BGEM3Embeddings  # noqa: E402
from app.services.rerank_service import rerank  # noqa: E402

settings = get_settings()

QA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "InduOCRBench", "RAG_eval", "QA_pairs.jsonl")
EVAL_CHROMA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_eval_ocr")
)
COLLECTION_NAME = "induocrbench_ocr_eval"
CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocr_md_cache")
)

FRIENDLY_CATEGORIES = {
    "Basic Recognition",
    "Structural Alignment",
    "Cross-Field Continuity",
    "Complex Reasoning",
}

# 生成答案 + RAGAS 判定的缓存（同题不重跑，省 LLM 调用）
GEN_CACHE = os.path.join(os.path.dirname(__file__), "ragas_gen_cache.jsonl")


def load_friendly_qa(sample_n: int | None = None) -> list[dict]:
    items = [json.loads(l) for l in open(QA_PATH, encoding="utf-8")]
    friendly = [it for it in items if it.get("question_category") in FRIENDLY_CATEGORIES]
    if sample_n and len(friendly) > sample_n:
        random.seed(42)
        friendly = random.sample(friendly, sample_n)
    return friendly


def _load_gen_cache() -> dict:
    cache = {}
    if os.path.exists(GEN_CACHE):
        for line in open(GEN_CACHE, encoding="utf-8"):
            try:
                it = json.loads(line)
                cache[it["q"]] = it
            except Exception:
                pass
    return cache


def _save_gen(question, retrieved, response):
    with open(GEN_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"q": question, "retrieved": retrieved, "response": response}, ensure_ascii=False) + "\n")


async def retrieve_top5(col, question: str) -> list[str]:
    vec = await encode_single(question)
    res = col.query(query_embeddings=[vec], n_results=settings.retrieve_top_k)
    docs = res["documents"][0]
    pairs = await rerank(question, docs, top_k=5)
    return [docs[i] for i, _ in pairs]


async def generate_samples(qa_items, col, gen_cache) -> list[SingleTurnSample]:
    """对每题：检索 → 生成答案 → 组装 SingleTurnSample。带缓存。"""
    samples = []
    n = len(qa_items)
    for idx, item in enumerate(qa_items):
        q = item["question"]
        ref = str(item["answer"])
        # 缓存命中
        if q in gen_cache:
            c = gen_cache[q]
            retrieved = c["retrieved"]
            response = c["response"]
        else:
            retrieved = await retrieve_top5(col, q)
            messages = _build_rag_prompt(q, retrieved)
            response = await chat(messages, thinking=False, max_tokens=512)
            _save_gen(q, retrieved, response)
            gen_cache[q] = {"q": q, "retrieved": retrieved, "response": response}
        samples.append(SingleTurnSample(
            user_input=q,
            response=response,
            retrieved_contexts=retrieved,
            reference=ref,
        ))
        if (idx + 1) % 5 == 0:
            print(f"    生成进度: {idx+1}/{n}")
    return samples


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("RAGAS 端到端评测（生成层：Faithfulness + Answer Relevancy）")
    print("=" * 60)

    qa_all = [json.loads(l) for l in open(QA_PATH, encoding="utf-8")]
    friendly = [it for it in qa_all if it.get("question_category") in FRIENDLY_CATEGORIES]
    random.seed(42)
    sample = random.sample(friendly, args.sample)
    cached = {f.replace(".md", "") for f in os.listdir(CACHE_DIR) if f.endswith(".md")}
    sample = [q for q in sample if q["filename"].replace(".md", "") in cached]
    print(f"有效题目: {len(sample)} 题\n")

    col = chromadb.PersistentClient(path=EVAL_CHROMA_DIR).get_collection(COLLECTION_NAME)
    gen_cache = _load_gen_cache()
    print(f"生成缓存: {len(gen_cache)} 题\n")

    # 1. 生成答案
    print("[1/2] 检索 + 生成答案...")
    samples = await generate_samples(sample, col, gen_cache)
    print(f"  生成 {len(samples)} 个样本\n")

    # 2. RAGAS 打分
    print("[2/2] RAGAS 打分（Faithfulness + Answer Relevancy）...")
    dataset = EvaluationDataset(samples)

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0,
    ))
    judge_emb = LangchainEmbeddingsWrapper(BGEM3Embeddings())

    # AnswerRelevancy 需要 embeddings。
    # strictness=1：默认 strictness=3 会用 n=3 多采样生成反推问题，
    # DeepSeek 不支持 n>1，必须降为 1（单次采样，仍能量化答案切题度）。
    metrics = [Faithfulness(), AnswerRelevancy(strictness=1)]

    # RAGAS evaluate 内部用 asyncio，需在 nest_asyncio 环境跑
    import nest_asyncio
    nest_asyncio.apply()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        show_progress=True,
        raise_exceptions=False,
    )

    print("\n" + "=" * 60)
    print("RAGAS 结果")
    print("=" * 60)
    # 0.4.3 返回 EvaluationResult，用 to_pandas() 取逐题分数再算均值
    scores = {}
    try:
        df = result.to_pandas()
        for m in ["faithfulness", "answer_relevancy"]:
            if m in df.columns:
                col = df[m].dropna()
                scores[m] = round(float(col.mean()), 4) if len(col) else None
                print(f"  {m}: {scores[m]}  (有效样本 {len(col)}/{len(df)})")
        print("\n逐题明细（前10题）:")
        for i, row in df.head(10).iterrows():
            f = row.get("faithfulness", "?")
            r = row.get("answer_relevancy", "?")
            fs = f"{f:.2f}" if isinstance(f, (int, float)) else str(f)
            rs = f"{r:.2f}" if isinstance(r, (int, float)) else str(r)
            print(f"  [{i}] faith={fs} rel={rs} | q={str(row.get('user_input',''))[:30]}...")
    except Exception as e:
        print(f"  结果读取失败: {e}")

    out = os.path.join(os.path.dirname(__file__), "eval_result_ragas.json")
    out_data = {"n": len(samples), "scores": scores}
    json.dump(out_data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    asyncio.run(main())
