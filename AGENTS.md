# AGENTS.md — DocQA（AI 智能文档问答系统）

私有文档知识库 RAG 系统：上传 PDF/DOCX/MD → 解析分块向量化 → 自然语言多轮提问 → 带来源引用的流式回答。FastAPI + LangGraph + ChromaDB + Redis + SQLite，后端托管静态前端（非前后端分离）。

## 开发准则

### 1. 动手前先想清楚

- 把需求复述为最小验收标准——复述不出就是还没理解
- 显式说出假设；不确定就问，别猜
- 多种理解并存时，列出来别默默选一个
- 有更简单的方案就说，该 push back 就 push back

### 2. 简单优先

- 只写解决问题的最小代码，不做投机性抽象
- 不为单次使用代码搞抽象层；没要求的"灵活性/可配置"不加
- 不为不可能的场景写错误处理
- 200 行能压到 50 行就重写
- 自检："senior 看会不会觉得过度设计？"

### 3. 外科手术式改动

- 只动必须动的，不顺手"改进"邻近代码/注释/格式
- 不重构没坏的东西；匹配现有风格，哪怕你觉得有更好的写法
- 发现无关死代码 → 提，不删
- 自己改动产生的孤儿（unused import/var）必须清
- 测试：每行改动都能追溯到用户需求

### 4. 目标驱动

- 把任务转成可验证目标：
  - "加校验" → "先写非法输入测试，再让它通过"
  - "修 bug" → "先写复现测试，再修到通过"
  - "重构 X" → "前后测试都过"
- 多步任务给简要 plan + 每步 verify

### 5. 代码 Review 顺序（做 review / 自检时）

1. 先确认主路径能跑、关键场景覆盖——没验证清楚优先指出
2. 评估是否当前上下文最优解；有更简洁但改动面更大的方案，先说明取舍别直接重写
3. 查过度设计/过度防御/过度嵌套：无关功能、用兜底掩盖设计缺陷、helper 过多调用链绕
4. 评估测试价值：只"给出靶子后评估靶子"的低价值测试建议清理，保留验证真实行为/关键路径/回归风险的

### 6. 不用过度防御掩盖设计缺陷

良好的软件在预设条件下运行；其余情况应及时报错/修复，不要靠冗余兜底代码掩盖问题。

## 功能开发流程

新功能**按规模分级**决定是否写 spec——避免"改个 prompt 也写文档"的形式主义（违反"简单优先"准则）。

### 需要写 spec（放 `spec/` 目录，参考 [`spec/TEMPLATE.md`](./spec/TEMPLATE.md)）

- 新增 Agent 节点 / 改 graph 拓扑（如加循环边、加条件路由）
- 新增工具调用 / 新增子 Agent
- 新增模块（新 service / 新 API 域 / 新解析器）
- 跨多文件重构
- 涉及架构取舍（如换检索架构、改记忆方案）

文件命名：`{日期}-{功能名}.md`（如 `20260706-web-search-tool.md`）。

### 不需要写 spec

- 改 prompt / 调阈值 / 修 bug / 加单个 API
- 补测试、改文档
- 调 UI 样式

这些直接动手，按"目标驱动"准则做即可。

## 常用命令

```bash
# 安装依赖（uv 管理，Python 3.12+）
uv sync

# 本地开发（首次启动加载 BGE-M3/Reranker 模型约 30s，常驻约 3GB 内存）
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
# 或直接用 venv 解释器绕过 uv run 的重装检查
.venv\Scripts\uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 测试（需 Redis 在 localhost:6379/0，conftest 会 flushdb）
.venv\Scripts\python -m pytest                       # 全量
.venv\Scripts\python -m pytest tests/test_cache.py -v # 单文件
.venv\Scripts\python -m pytest tests/test_xxx.py::test_func -v  # 单用例

# GPU 加速（可选，BGE-M3 encode 21.5s→0.2s）
uv add "torch==2.12.0+cu126" --index-strategy unsafe-best-match

# Docker 部署（app + redis 双服务，单 worker——模型常驻内存多 worker 重复占 3GB）
docker compose up -d --build
```

启动前：Redis 必须运行；`.env` 参考 `.env.example`（必填 `LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL`）；模型权重放 `models/bge-m3`、`models/bge-reranker-v2-m3`。

## 架构与分层边界

```
app/
├── api/        路由层：auth / documents / chat（HTTP 边界，参数校验，调 service）
├── services/   业务层：document_service / chat_service / llm_provider /
│               embedding_service / rerank_service / text_splitter /
│               summary_service / ragas_embed_adapter
├── agent/      LangGraph 状态机：graph(编排) / nodes(节点) / state(状态) / memory(记忆工具)
├── core/       基础设施：cache(redis) / database(sqlite) / security(jwt) /
│               rate_limit(slowapi) / response(统一响应码) / exceptions
├── models/     SQLAlchemy ORM：user / document / conversation+message
└── schemas/    Pydantic 请求/响应模型（API 边界校验，与 ORM 分离）
```

**分层规则**：
- `api/` 只做 HTTP 编排，业务逻辑放 `services/`，不直接操作 DB/向量库
- `agent/` 的节点函数（nodes.py）编排检索+生成；记忆/缓存/持久化织入在 api 层和 service 层，**不是 LangGraph 节点**
- `models/`（ORM 持久化）与 `schemas/`（API 边界）严格分离，勿混用
- 流式生成走手动编排（`chat_service.stream_graph`），**不用 `graph.astream_events`**（OpenAI SDK 直连后该事件不触发）

## 关键约定

**配置**：所有参数集中在 `app/config.py`（pydantic-settings，`.env` 覆盖）。模块顶层 `settings = get_settings()` 绑定单例——测试改配置必须 `patch.object(module.settings, attr, val)`，改 env + 清 lru_cache 对已导入模块无效。

**统一响应**：所有 API 返回 `{code, message, data}`（`code=0` 成功）。错误码在 `app/core/response.py` 按前缀分组（10xxx 认证 / 20xxx 文档 / 30xxx 对话 / 40xxx 通用）。业务异常 `raise BizError(code=..., message=...)`，全局 handler 统一处理，勿手动构造错误 JSON。

**LLM 调用**：`app/services/llm_provider.py` 封装 OpenAI SDK 直连 DeepSeek。`chat()` 非流式、`astream_chat()` 流式。thinking 开关显式控制（`{"thinking":{"type":"disabled"}}`），`deepseek-v4-flash` 默认开 thinking 需显式 disabled。**不要用 langchain-deepseek**（历史遗留依赖，已被 OpenAI SDK 方案取代）。

**向量库**：ChromaDB 按用户隔离 collection（`doc_user_{user_id}`），所有读写加 `async with _chroma_lock`。chunk metadata 含 `content_hash`（增量更新 diff 用）。删除文档用 `collection.delete(where={"document_id": id})`。

**缓存 key 规范**：`qa:user_{uid}:conv_{cid}:{问题哈希}:h{历史版本}`（问答缓存，TTL 30min）/ `history:conv_{cid}`（历史缓存，每轮清）/ `lock:...`（互斥锁）。问答缓存 key 含 history_version 段（用历史消息条数），防止多轮追问命中过期答案。

**异步文档处理**：上传走 BackgroundTasks（`create_pending_document` 建 pending 记录 → `process_pending_document` 后台处理），状态流转 pending→processing→done/failed。`replace_id` 走增量更新（`update_document_chunks`，chunk hash diff + 变化率超阈值降级全量）。

**Python 风格**：符合 pythonic；用 3.12+ 语法（如 `X | None` 而非 `Optional[X]`、`list[str]` 而非 `List[str]`）；不允许把代码写稀碎——不为线性逻辑拆细 helper，拆函数只服务于明确的复用、隔离副作用或降低认知负担；遵循向下规则（公开方法在文件顶部，细节逐层下沉，读者从上到下不跳跃）。

## 测试

- 测试用真实 Redis + 真实 SQLite + 真实 ChromaDB（fixture 区分、ChromaDB 清理坑、mock LLM 等**详见 [testing-guidelines.md](./testing-guidelines.md)**）
- **关键坑**：`_db` fixture 不自动清 ChromaDB——测 ChromaDB 的用例必须自己删 `data/chroma_test` 目录 + 重置 `_chroma_client` 单例，否则跨用例残留污染
- **mock LLM**：patch 节点文件里 import 的名字（`patch("app.agent.nodes.chat", ...)`），不是 `llm_provider.chat`——节点捕获的是本地引用，patch 源头对已导入模块无效
- 评测脚本在 `tests/eval/`（检索 Hit Rate/MRR + RAGAS 端到端），数据集 `InduOCRBench/`（gitignored，826MB）

## 平台与部署注意

- **Windows 开发**：Bash 命令用 Git Bash；`cd /d` 在 bash 里失败，用 `cd "D:/path"`；`.venv\Scripts\` 而非 `.venv/bin/`
- **Docker 生产**：单 worker（`gunicorn -k uvicorn.workers.UvicornWorker -w 1 --preload`），多 worker 重复占模型内存；`pyproject.toml` 的 `[tool.uv.sources] torch` 行注释掉回退 CPU 版；Nginx 需 `proxy_buffering off`（SSE）
- **数据目录**：`data/`（SQLite+ChromaDB+uploads，gitignored）、`models/`（模型权重，gitignored）不入库

## 文档（改敏感区域前先读）

- `README.md` — 项目总览、特性、快速开始、API
- `rag指标记录.md` — 5 个评测结果（检索层 94%/RAGAS 0.78 等）
- 笔记目录 `D:\note\ck\project\个人项目\文档知识库问答\` 下：
  - `项目中的困难与决策.md` — 各 stage 决策记录（stage-5~14）
  - `DocQA可能会被问的问题.md` — 面试 Q&A（Q0~Q51）
  - `项目学习文档/` — 各技术点学习文档（记忆/缓存/分块/检索/SSE 等）
  - `项目不足与教训.md` — 踩坑记录
