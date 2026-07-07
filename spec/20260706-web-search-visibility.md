# Spec：联网搜索可见性

> 依据 `spec/TEMPLATE.md` 的方案级规格编写。目标是让用户在前端和历史记录中明确感知“本次回答是否参考了联网搜索结果”。

## 背景

当前 `stream_graph` 已经支持联网搜索链路：CRAG 重查后仍低相关时，如果 `ENABLE_WEB_SEARCH=true`，会调用 Tavily 获取网络内容并构造 `_build_web_search_prompt`。

但用户无法感知联网搜索是否触发：
1. 前端只渲染 `sources`，无“联网搜索”提示；
2. assistant 消息落库无联网搜索标记，历史记录无法回溯；
3. 若联网搜索失败，用户仍不知道，只能从回答内容推断。

该缺失导致用户无法验证联网搜索是否生效，也不利于调试与信任建立。

## 目标

- [ ] 用户在联网搜索回答的气泡中可看到“本次回答参考了联网搜索结果”提示；
- [ ] 历史消息回放时仍保留该可见性标记（不因刷新丢失）；
- [ ] 若联网搜索未触发或触发失败，前端不显示误导信息；
- [ ] 不影响现有 SSE 主链路与流式性能；
- [ ] 不改变已有的 `sources` 渲染语义（联网搜索结果与文档来源分离呈现）。

## 方案

### 设计思路

1. 在 `stream_graph` 中联网搜索触发时，新增 SSE 事件 `web_search`，携带 `{used: true, query: ..., results_preview: ...}`；
2. 将“是否使用联网搜索”落库到 `Message` 中，优先存在 `meta` 字段（如 `{"web_search_used": true}`），便于历史回放；
3. 前端在 `chat.js` 中监听 `web_search` 事件，给 assistant 气泡追加 chip 提示（如“联网搜索”）；
4. 读历史时读取 `Message.meta`，若 `web_search_used=true`，同样显示该 chip。

### 涉及改动

- `app/services/chat_service.py`
  - `_try_web_search()` 返回值补充更多元信息（是否成功、query、结果数量）；
  - `stream_graph()` 在触发联网搜索后 `yield ("web_search", payload)`；
  - 落库时将 `meta.web_search_used` 写入 Message。
- `app/api/chat.py`
  - `event_stream()` 监听 `web_search` 事件并透传给前端；
  - `_finalize()` 持久化联网搜索标记到 Message。
- `static/js/chat.js`
  - 新增 `web_search` 事件处理，给 assistant 气泡加 chip/提示；
  - 历史渲染时读 `meta`，决定是否显示提示。
- `tests/test_tool_calling.py`
  - 新增测试：联网搜索触发时 stream_graph 输出 `web_search` 事件；
  - 新增测试：未触发/失败时不输出 `web_search` 事件。

### 关键决策与取舍

- 决策 1：新增 `web_search` 事件，而不是复用 `sources`。原因是联网搜索结果与文档来源应分离，避免语义混淆。
- 决策 2：用 Message.meta 存联网搜索标记，而不是新字段。原因是可扩展、兼容现有消息模型，减少 schema 变更。
- 决策 3：在 `_try_web_search()` 层面返回元信息，而不是让 stream_graph 再调一次 Tavily。原因是避免重复请求、保持单一职责。

## 验收标准

- [ ] 知识库无相关内容 + CRAG 失败 + 开启联网搜索 → 前端显示“联网搜索”提示；
- [ ] 回顾历史消息，联网搜索回答仍显示提示；
- [ ] 若联网搜索未触发（如相关文档命中、general 路径、Tavily key 缺失），不显示提示；
- [ ] 测试通过：`tests/test_tool_calling.py` + `tests/test_chat.py`。

## 非目标

- 不实现联网搜索结果的独立来源面板（只做 chip 提示）；
- 不在前端展示 Tavily 返回的原始 URL 列表（保持简洁）；
- 不实现联网搜索结果的缓存/去重（当前每次低相关都重新搜索）。

## 参考

- `app/services/chat_service.py`（当前联网搜索链路）
- `app/services/tools/web_search.py`（Tavily 调用）
- `static/js/chat.js`（SSE 事件渲染）
- `spec/20260706-tool-calling.md`（工具调用设计）

## 进度

- [ ] spec 评审通过
- [ ] 后端实现联网搜索事件与落库标记
- [ ] 前端显示联网搜索提示
- [ ] 测试覆盖
