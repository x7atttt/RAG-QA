# Spec：文档引用说明（Document Citation Context）

## 背景

RAG 系统有一个天然盲区：**meta-level 问题**（关于文档的问题）和文档内容语义方向不同。

典型场景：用户上传了"个人简历.docx"，然后问"我的简历有什么建议"。简历 chunk 的内容是"姓名:李宇 电话：15360252519"，query 是"简历建议"——reranker 给出 score=0.10（阈值 0.5），简历被切掉，LLM 只拿到综合篇.pdf 的面试建议，回答"文档未直接涉及"。

**根因**：query "我的简历有什么建议" 和 chunk "姓名:李宇 电话..." 在向量空间里距离远。chunk 内容描述的是"我有什么"，query 问的是"怎么改"。BGE-M3 和 BGE-Reranker 都无法建立这种语义关联。

**价值**：解决后，用户问任何关于文档的 meta-question（"我的简历怎么样""总结一下这篇报告""这个方案有什么问题"）都能正确关联到对应文档并引用来源。

## 目标

- [ ] 目标 1：用户问"我的简历有什么建议"时，检索能命中简历 chunk → 验证：rerank score ≥ 0.5 或 fallback 路径 sources 包含"个人简历.docx"
- [ ] 目标 2：引用说明由 LLM 自动生成，用户无需手动填写 → 验证：上传文档后 Document 表有 summary 字段
- [ ] 目标 3：已有文档重新入库后也生效 → 验证：手动触发 re-index 后检索命中

## 方案

### 设计思路

**核心思路：分块时在每个 chunk 前拼接一段文档引用说明，让 chunk 内容包含文档的高层描述。**

```
原始 chunk:
  "姓名:李宇 电话：15360252519 邮箱：yu.li2027@qq.com..."

拼接引用说明后:
  "[来源: 个人简历] 这是一份Python开发者的个人简历，包含教育背景、技术栈、项目经历和获奖情况。
  姓名:李宇 电话：15360252519 邮箱：yu.li2027@qq.com..."
```

这样 "我的简历" 和 chunk 的相似度大幅提升——chunk 里有"简历"这个词了。

**为什么在入库时拼接而不是检索时拼接**：
- 入库时拼接 → summary 成为 embedding 的一部分 → 向量本身就包含文档描述信息 → 检索时自然命中
- 检索时拼接 → 需要额外查 DB 取 summary → 改动链路更长，且 reranker 仍可能判低分

**为什么用 LLM 生成而不是用文件名**：
- 文件名太短（"个人简历.docx"），信息量不足
- LLM 能从文档内容提取关键主题（"Python开发者简历，包含教育背景、技能清单、项目经历"），语义更丰富
- 一次生成，所有 chunk 复用，成本可控

### 涉及改动

- `app/models/document.py`：Document 模型加 `summary` 字段（TEXT, nullable）
- `app/core/database.py`：`_ensure_column` 迁移补列
- `app/services/document_service.py`：`process_pending_document` 中解析后、分块前，调 LLM 生成 summary；分块时拼接到每个 chunk 前面
- `app/services/llm_provider.py`：新增 `chat` 调用（复用现有函数，不需要新接口）
- `tests/test_document_summary.py`：测试 summary 生成 + chunk 拼接逻辑

### 关键决策与取舍

- **决策 1：summary 拼到 chunk 前面 vs 存为独立 chunk**
  - 选拼接：独立 chunk 检索到后只是文档描述，没有具体内容，喂给 LLM 信息量不够。拼接方式让每个 chunk 都携带文档上下文
  - 代价：每个 chunk 多 ~50 token（2-3 句话），对 500 字符的 chunk 约增加 10%，embedding 向量略有稀释，但 summary 是高质量信号，利远大于弊

- **决策 2：生成 summary 的时机**
  - 选入库时（process_pending_document）：一次生成，后续检索零成本
  - 不选检索时：每次查询都要额外调 LLM，延迟增加，且 CRAG 循环会多次触发

- **决策 3：summary 长度控制**
  - 限制 2-3 句话（~100 字），prompt 显式要求"简洁描述文档主题和内容类型"
  - 太长会稀释 chunk 内容的 embedding 信号

## 验收标准

- [ ] 主路径：上传"个人简历.docx" → 问"我的简历有什么建议" → sources 包含"个人简历.docx" → 回答基于简历内容给建议
- [ ] 主路径：上传文档后 DB 中 Document.summary 非空，内容为 2-3 句话的文档描述
- [ ] 边界：summary 生成失败 → 降级用文件名作为 summary，不阻断入库流程
- [ ] 边界：chunk 拼接后长度超过 chunk_size → 不额外截断（summary 是必要上下文，宁可超一点）
- [ ] 兼容：已有文档（无 summary）检索行为不变（拼接为空字符串，等效于现状）
- [ ] 测试：test_document_summary.py 覆盖 summary 生成、chunk 拼接、降级逻辑

## 非目标

- 不做用户手动填写引用说明（全自动，零配置）
- 不做跨文档关联（如"我的简历和课程设计有什么关联"）
- 不改检索管道本身（dense/sparse/rerank 参数不变）
- 不改 CRAG 循环逻辑

## 参考

- 项目 `app/services/document_service.py` — 文档入库流程
- 项目 `app/services/text_splitter.py` — 分块策略
- 项目 `app/models/document.py` — Document 模型
- 项目 `app/agent/nodes.py` — retrieve_documents 检索逻辑

---

## 进度（实现时更新）

- [ ] 设计评审
- [ ] 实现
- [ ] 测试
- [ ] 文档（README / 学习文档 / 面试 Q&A / 决策记录）
