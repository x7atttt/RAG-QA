# Spec：对话中断与继续回答

> 状态：设计中 · 创建于 2026-07-07 · 参考 [`spec/TEMPLATE.md`](./TEMPLATE.md)

## 背景

当前流式问答是"发了就不能停"的单向管道：用户点击发送后，SSE 流一直跑到结束，期间没有任何手段中断。`AbortController` 仅用于 60s 超时和页面卸载，用户无法主动停止生成。

这带来两个实际问题：
1. **用户体验差**：LLM 已经在跑偏或生成冗余内容时，用户只能干等，无法及时止损
2. **部分答案丢失**：如果用户中途离开页面（`beforeunload` 触发 abort），后端虽然通过 `asyncio.shield(_finalize(...))` 保存了已生成的部分答案，但没有标记为"不完整"——前端无法区分这是完整回答还是中断残片，也无法提供"继续回答"的入口

这是面试高频考点：SSE 流式传输、优雅断开、async generator 取消、部分状态持久化、流恢复——生产级系统的典型问题。

## 目标

- [x] **停止按钮**：流式回答进行中，用户可随时点击"停止"中断生成 → 验证：点击停止后 SSE 流中断，已生成内容保留
- [x] **部分答案持久化**：中断的答案标记为 `partial`，落库不丢失 → 验证：DB 中 `messages.status = 'partial'`，刷新页面后能看到
- [x] **继续回答**：对 partial 消息可点"继续回答"，从断点续写 → 验证：续写内容追加到同一气泡，完成后 status 变 `complete`
- [x] **历史兼容**：加载历史消息时 partial 消息显示继续按钮 → 验证：切换会话再切回来，partial 消息仍可继续
- [x] **不破坏现有主链路**：正常完成的回答 status 仍为 `complete`，现有测试不回归 → 验证：`pytest tests/test_chat.py` 全过

## 方案

### 设计思路

参考 ChatGPT / Claude 的标准模式：**停止 → 保留部分内容 → 标记不完整 → 可继续**。

核心流程：

```
用户点"停止"
  → 前端 AbortController.abort() 中断 SSE
  → 后端 CancelledError → finally → _finalize(is_partial=True) 保存部分答案
  → 前端显示"⏹ 已停止" + "继续回答"按钮

用户点"继续回答"
  → 前端 POST /api/chat/continue { message_id, conversation_id }
  → 后端加载 partial 消息 + 上下文，构造续写 prompt
  → astream_chat 流式续写，token 追加到同一气泡
  → 完成后 UPDATE 原 message：content 合并，status → complete
```

### 涉及改动

**模型层**：
- `app/models/conversation.py` — Message 加 `status: str` 字段（`complete` / `partial`，默认 `complete`）
- `app/core/database.py` — `init_db` 加 `_ensure_column` 轻量迁移

**Schema 层**：
- `app/schemas/chat.py` — `MessageOut` 加 `status` 字段；新增 `ChatContinueRequest`

**API 层**：
- `app/api/chat.py`：
  - `_finalize` 加 `is_partial` 参数，存 Message 时写入 status
  - `event_stream` 捕获 `CancelledError` 时设 `partial = True`
  - 新增 `POST /continue` 端点：加载 partial 消息 → 构造续写 prompt → `astream_chat` 流式生成 → UPDATE 合并

**前端**：
- `static/chat.html` — 加停止按钮（`#stopBtn`，默认 hidden）
- `static/js/chat.js`：
  - `setSending` 切换 sendBtn/stopBtn 可见性
  - stopBtn click → abort + 标记 `userStopped`
  - AbortError catch 区分用户停止 vs 超时：用户停止显示"继续回答"按钮
  - 继续按钮 click → POST `/continue`，token 追加到现有 contentEl
  - 历史渲染：`status === 'partial'` 的消息显示继续按钮

**测试**：
- `tests/test_continue.py`（新）：partial 保存、continue 端点、合并逻辑

### 关键决策与取舍

**决策 1：Message 模型加 `status` 字段，不在 content 里插标记**
- 选 A（字段）不选 B（content 尾部加 `\u200B` 标记）。字段方案语义清晰、查询方便、不污染内容。content 标记方案虽然不用改模型，但需要前端/后端都做 trim 处理，且历史数据兼容麻烦。

**决策 2：继续时不重新跑检索链路（CRAG），复用首次检索的 sources**
- 选 A（复用）不选 B（重跑 pipeline）。继续的语义是"接着说"，不是"重新回答"。重跑 CRAG 可能得到不同 chunk，导致续写内容与前半部分风格/信息不一致。复用 sources 需要在 Message 模型中已有 sources 字段（已有），从 sources 反推 prompt 类型（RAG / web / fallback）构造续写上下文。
- 取舍：如果首次检索的 sources 不理想，续写也只能基于它们。用户若不满意可以重新提问。

**决策 3：继续端点用 `UPDATE` 合并到原 message，不新建 message**
- 选 A（合并）不选 B（新建 message）。合并后用户看到一条完整消息，体验自然。新建 message 会导致同一轮回答分裂为两条，历史渲染和缓存逻辑都需要适配。
- 取舍：合并需要 `UPDATE` 操作，比 `INSERT` 稍复杂；续写过程中如果再次中断，需要处理"partial + 已有部分内容"的叠加状态。

**决策 4：前端停止后不发额外请求标记 partial，靠后端 CancelledError 自动处理**
- 前端 abort → SSE 断开 → 后端 CancelledError → `_finalize(is_partial=True)`。不需要前端额外调接口标记。简化流程，避免竞态（abort 后立刻标记，但后端可能还没执行完 finalize）。

## 验收标准

**主路径**：
- [ ] 流式回答中点"停止" → 已生成内容保留，显示"⏹ 已停止"+继续按钮
- [ ] 点"继续回答" → 从断点续写，追加到同一气泡，完成后 status = `complete`
- [ ] 正常完成的回答 → status = `complete`，不显示继续按钮

**边界 / 异常**：
- [ ] 中途离开页面 → 部分答案保存为 partial，刷新后能看到 + 可继续
- [ ] 历史加载 partial 消息 → 显示继续按钮，点击后正常续写
- [ ] 续写过程中再次点停止 → 部分续写内容也正确保存
- [ ] 对 complete 消息点继续 → 不应出现（前端不显示按钮），后端兜底拒绝
- [ ] 并发：停止后立即发新问题 → 不冲突（streaming flag 正确重置）

**兼容性**：
- [ ] 现有测试 `tests/test_chat.py` 全过
- [ ] 旧数据（无 status 字段）→ 默认 `complete`，行为不变

## 非目标

- **不做**流式生成速度控制（如"慢速输出"）
- **不做**多分支回答（如"生成 3 个版本让用户选"）
- **不做**回答编辑后重新生成（这是另一个功能）
- **不做**服务端主动超时中断（当前 60s 超时由前端控制，保持不变）

## 参考

- ChatGPT 的 Stop generating / Continue generating 交互模式
- Claude 的流式中断与续写行为
- FastAPI SSE `CancelledError` 处理：`asyncio.shield` 保护落库逻辑
- 项目既有决策：`_finalize` 的 `asyncio.shield` 设计（保护断连后数据不丢）

---

## 进度（实现时更新）

- [x] 设计评审
- [x] 实现（2026-07-07）
- [x] 测试（8 passed，test_continue.py）
- [x] 文档（README / 面试 Q&A Q62-Q65 / 决策记录 Stage-17 / 流程图）
