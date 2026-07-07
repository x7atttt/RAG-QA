# Spec：工具调用（联网搜索 + 文档元信息查询）

> 状态：设计中 · 创建于 2026-07-06 · 参考 [`spec/TEMPLATE.md`](./TEMPLATE.md)

## 背景

当前 Agent 是"检索 → 生成"的线性 + CRAG 循环结构，存在两类**真实能力缺口**：

**缺口 ①：CRAG 失败后无外部信息源**
`stream_graph` 的 CRAG 循环在 `top_score < 阈值且达重查上限` 时 break，走 `_build_fallback_prompt`——**仅凭 LLM 常识 + 残余弱相关片段**作答。但很多场景（"2026 新政策""文档库没收录的术语"）真正需要的是**外部信息**，现在只能编（幻觉）或拒答。

**缺口 ②：元层问题彻底答不了**
检索是 **chunk 级语义匹配**，对"我上传过哪些 PDF""最近一周加了什么文档""哪份最长"这类**文档元信息问题**完全失效——这些问题的答案不在 chunk 里，而在文档元数据里。当前架构对这类问题要么硬塞无关 chunk（fallback 降级），要么拒绝。

工具调用正好补这两类缺口：每个工具补一种"现有架构答不了"的问题类型。

## 目标

- [ ] **联网搜索工具**：CRAG 失败分支能调 Tavily，回答含外部信息 → 验证：构造知识库无相关内容的问题，确认答案引用了网络内容（来源标注区分"文档"vs"网络"）
- [ ] **文档元信息工具**：LLM 自主决策调用，答元层问题 → 验证：问"我上传过哪些 PDF""最近的文档"，工具被调用且返回正确元信息
- [ ] **不破坏现有主链路**：CRAG 循环、流式输出、记忆织入（摘要 + 历史预算）行为不变 → 验证：现有测试（test_crag / test_memory / test_chat）不回归
- [ ] **现有评测不回归**：检索层 Hit@3 94% / RAGAS Faithfulness 0.78 → 验证：重跑 tests/eval/ 指标稳定

## 方案

### 设计思路：两种范式混合

两个工具的**触发性质不同**，所以用不同范式——不是为了展示两种范式而硬分，是工具本身的性质决定的：

| 工具 | 触发性质 | 范式 | 理由 |
|------|---------|------|------|
| 联网搜索 | **条件触发**（CRAG 失败信号明确）| 规则触发 | LLM 决策反而有风险——LLM 可能因"自以为能答"而不调用，导致 CRAG 失败时该搜不搜。规则触发保证必触发 |
| 文档元信息 | **问题类型触发**（要靠语义判断）| LLM 决策（bind_tools）| 规则触发做不到——"我有哪些文档"和"文档里讲了什么"用词接近，硬规则（如关键词匹配）会误判。必须靠 LLM 理解问题类型 |

### 涉及改动（位置清单，不含实现）

**新增**：
- `app/services/tools/` 目录——工具实现集中在此
  - `web_search.py`——Tavily 封装（输入 query，输出结果片段 + URL）
  - `doc_meta.py`——文档元信息查询（输入查询意图，输出文档列表/统计，从 SQLAlchemy 查）
- `app/agent/nodes.py` 加 `tool_router` 节点（LLM 决策"这问题要不要调文档元信息工具"）

**改动**：
- `app/services/chat_service.py` 的 `stream_graph`——在两处加挂点：
  - **挂点 ①**（intent_router 之后）：调 `tool_router`，若决定调文档元信息工具则走工具分支（跳过检索链路）
  - **挂点 ②**（CRAG 循环 break 之后）：若 `top_score < 阈值` 则调 `web_search`，结果作为额外 context 进 `_build_fallback_prompt`（或新增 `_build_web_search_prompt`）
- `app/config.py`——加 `TAVILY_API_KEY`、`ENABLE_WEB_SEARCH` 开关（默认关，没 key 不影响主流程）
- `app/agent/state.py`——AgentState 加字段（如 `tool_call` / `external_context`，命名实现时定）
- `.env.example`——补 `TAVILY_API_KEY=`

**不改**：
- `intent_router` 不动（保持现有"是否检索"的职责，tool_router 串在它**之后**）
- `retrieve_documents` / `grade_documents` / `transform_query` / `rewrite_query` 不动
- 流式生成（`astream_chat`）逻辑不动
- 记忆织入（`_load_recent_history` / `_load_summary` / `_finalize`）不动

### 关键决策与取舍

**决策 1：新加 `tool_router` 节点，不升级 `intent_router`**
- 选 B 不选 A。`intent_router` 现在做两件事（无文档短路 + LLM 二分检索意图），职责已接近饱和；再塞"是否调文档元信息工具"会让它判断三件事，prompt 难写且易误判。
- 代价：多 1 次 LLM 调用。但 `tool_router` 只在 `intent_router` 通过后跑（无文档 / 纯闲聊场景提前短路），高频问题多 1 次调用的成本可接受。
- 顺序：`intent_router → tool_router → 检索链路 / 工具分支 / 纯对话`，每节点一件事，符合"职责分离"。

**决策 2：联网搜索走规则触发，不交给 LLM 决策**
- CRAG 失败是**确定性信号**（`top_score < 阈值`），规则触发保证"该搜必搜"。
- LLM 决策在此场景有道德风险：LLM 可能因"我能凭常识答"而不调工具，导致 CRAG 失败时还是走幻觉 fallback。
- 代价：失去"LLM 主动判断要不要搜"的灵活性，但这个灵活性在 CRAG 失败的具体场景里是负价值。

**决策 3：文档元信息走 LLM 决策，不交给规则触发**
- 元层问题与内容检索问题**用词高度重叠**（"我有哪些文档" vs "文档里讲了什么"），硬规则（关键词匹配）误判率高。
- 必须靠 LLM 理解问题**类型**（meta vs content），这正是 `bind_tools` 的标准用法。
- 代价：1 次 LLM 调用 + 偶发误判（LLM 把元层问题误判为检索问题）。误判的代价低（顶多走检索 fallback，答得不准但不会崩）。

**决策 4：不引入 ToolNode + 条件边（不上 graph.invoke）**
- 现有 `stream_graph` 是手动编排，已弃用 `graph.astream_events`（OpenAI SDK 直连后该事件不触发）。引入 ToolNode 要改回 `graph.invoke`，等于**撤销 stage-8 的关键决策**，且重新踩"流式事件不触发"的坑。
- 工具调用在 `stream_graph` 手动编排里直接 `await tool()` 即可，不需要 ToolNode 抽象。
- 代价：失去 LangGraph 工具调用生态（如工具调用可视化），但本项目规模用不上。

**决策 5：联网搜索结果与文档检索结果分离，不混入 sources**
- 文档 sources 有溯源价值（用户可点开看原文 chunk），网络搜索结果可信度和粒度不同。
- 分离呈现（如新增 `web_results` 事件 / 在 sources 里标 `"source": "web"`），让用户能区分答案来源是文档还是网络。
- 代价：前端要改 sources 渲染逻辑，区分两种来源。

## 验收标准

**主路径**：
- [ ] 知识库无相关内容 + CRAG 失败 → 自动调 Tavily，答案引用网络内容，来源标注区分"网络"
- [ ] 问"我上传过哪些 PDF" → `tool_router` 调文档元信息工具，正确列出文件名/数量
- [ ] 问"最近一周加了哪些文档" → 工具被调用，按时间过滤返回

**边界 / 异常**：
- [ ] `TAVILY_API_KEY` 缺失 → 降级走原 fallback（不报错，仅日志）
- [ ] Tavily 调用失败 / 超时 → 降级走原 fallback（工具异常不阻断主流程）
- [ ] 文档元信息工具查询出错 → 降级走检索链路（不让工具失败阻断回答）
- [ ] `ENABLE_WEB_SEARCH=False` → 完全跳过联网搜索分支，行为同改造前

**不回归**：
- [ ] `pytest tests/` 全量通过
- [ ] 检索层 Hit@3 ≥ 94%、RAGAS Faithfulness ≥ 0.78（重跑 tests/eval/）

## 非目标

明确**不做**的事，防止范围蔓延：

- ✗ **多 Agent 协同**（已推迟——本 spec 只做单 Agent + 工具调用）
- ✗ **ToolNode + 条件边**（不重构 graph.invoke，理由见决策 4）
- ✗ **SQL 聚合工具**（个人知识库场景弱，且与 NL2SQL 项目能力重叠）
- ✗ **文档管理工具**（删除/打标签——改库有风险，API 已有不必接 agent）
- ✗ **工具调用历史 / 多轮工具调用记忆**（先做单轮工具调用）
- ✗ **MCP 集成**（参考 Yuxi 有，但个人项目用不上，徒增复杂度）
- ✗ **代码沙箱 / 画图工具**（场景弱）
- ✗ **多工具并行调用**（第一版两个工具性质不同不会并行）

## 参考

- `reference/open_deep_research-main/`——Tavily 工具实现参考（重点看它的 search tool 封装）
- `reference/Yuxi-main/docs/agents/tools-system.md`——`@tool` 装饰器 + 分类注册机制参考（不照抄架构，只学工具组织方式）
- `reference/Yuxi-main/docs/agents/mcp-integration.md`——MCP 集成方式（**本 spec 明确不做**，仅留作后续演进参考）
- LangGraph 官方 Agentic RAG 教程 https://docs.langchain.com/oss/python/langgraph/agentic-rag ——retriever 包成 tool 的最小实现
- 项目内相关：
  - `app/services/chat_service.py` 的 `stream_graph`——本 spec 主要改动位置
  - `app/agent/nodes.py` 的 `intent_router` / `transform_query`——挂点 ① 和挂点 ② 的相邻节点
  - 笔记 `D:\note\ck\project\个人项目\文档知识库问答\项目学习文档\CRAG循环重查.md`——挂点 ② 的衔接基础

## 进度

- [x] 设计讨论（4 个核心问题决策完成）
- [x] spec 草稿
- [ ] spec 评审（与自己对齐后定稿）
- [ ] 实现：Tavily 工具封装
- [ ] 实现：文档元信息工具
- [ ] 实现：tool_router 节点 + 挂点 ①
- [ ] 实现：CRAG break 后挂点 ②
- [ ] 测试：新增工具单测 + 集成测试
- [ ] 评测回归：tests/eval/ 重跑
- [ ] 文档：README 工具调用章节 / 学习文档 / 决策记录 / 面试 Q&A
