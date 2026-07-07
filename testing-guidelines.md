# 测试规范

## 测试基础设施

- 真实三方依赖：Redis（`localhost:6379/0`，conftest autouse flushdb）/ SQLite（`data/test.db`）/ ChromaDB（`data/chroma_test`）
- 不用 mock 替代存储层——真实依赖才能暴露集成问题（迁移、约束、向量库元数据等）

## 两种 fixture

| fixture | 用途 | ASGI | ChromaDB 自动清理 |
|---------|------|------|------------------|
| `client` | HTTP 集成测试 | 起 ASGITransport | 是（删 `data/chroma_test` 目录）|
| `_db` | DB 直连单元测试 | 不起 ASGI | **否**——需手动清 |

## ChromaDB 清理坑（跨用例残留污染）

`_db` fixture 不自动清 ChromaDB。测 ChromaDB 的用例（如 `test_incremental_update`）必须：

1. fixture 前后删 `data/chroma_test` 目录
2. 重置 `_chroma_client` 单例

否则跨用例残留污染（旧 collection / 旧 chunk 串进新用例）。

## mock LLM

patch 节点文件里 import 的名字，不是 `llm_provider.chat`：

```python
patch("app.agent.nodes.chat", new_callable=AsyncMock)
```

理由：节点函数捕获的是 `from app.services.llm_provider import chat` 的本地引用，patch 源头对已导入的模块无效。

## 测试目录组织

- `tests/test_*.py` — 单元/集成测试（按功能模块：auth / cache / chat / chunking / crag / documents / embedding / memory / retrieval / summary 等）
- `tests/conftest.py` — 共享 fixture（`client` / `_db`）
- `tests/eval/` — 评测脚本（检索 Hit Rate/MRR + RAGAS 端到端）

## 评测数据集

`InduOCRBench/`（gitignored，826MB）——中文企业文档评测集（12 行业 / 570 份 PDF / 2071 题）。
