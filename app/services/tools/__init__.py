"""工具调用模块。

工具分两类（spec/20260706-tool-calling.md）：
- web_search：联网搜索（Tavily）——规则触发（CRAG 失败兜底）
- doc_meta：文档元信息查询——LLM 决策触发（tool_router 路由）

工具实现集中在此目录，编排逻辑在 chat_service.stream_graph。
"""
