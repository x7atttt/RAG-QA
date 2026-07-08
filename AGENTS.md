# DocQA 项目协作规则

## 1. 环境与命令

### 虚拟环境（绕过 uv，直接用 .venv）

项目用 uv 管理依赖，但 uv 在国内镜像源经常 403。**所有命令直接用 `.venv` 虚拟环境**，不走 `uv run`。

```bash
# 启动开发服务器
.venv/Scripts/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 运行测试
.venv/Scripts/python -m pytest tests/ -v

# 运行单个测试文件
.venv/Scripts/python -m pytest tests/test_chat.py -v

# pip 安装（在 .venv 内）
.venv/Scripts/pip install <package>
```

### 前端静态文件

前端是纯 HTML + JS + CSS（无构建工具），改完刷新浏览器即可。**加版本号防缓存**：
```html
<script src="/static/js/chat.js?v=YYYYMMDD"></script>
<link rel="stylesheet" href="/static/css/style.css?v=YYYYMMDD">
```

### 前端 UI 风格：墨韵书卷

主题定位：**个人知识助手**——深靛蓝 + 暖米白 + 琥珀金，书卷气。

**Design Tokens**（`style.css :root`，所有颜色从这里取，不硬编码）：

| Token | 值 | 用途 |
|-------|-----|------|
| `--ink` | `#1e293b` | 主文字色、导航栏背景 |
| `--paper` | `#faf8f5` | 页面底色 |
| `--parchment` | `#f5f0e8` | AI 气泡背景、侧边栏 hover |
| `--amber` | `#d97706` | 强调色（按钮、链接、开关、光标） |
| `--border` | `#e8e0d4` | 边框、分割线 |

**字体**：
- 标题（`h1~h3`、`.navbar-brand`、`.fw-bold`）：`'Noto Serif SC'` 衬线，Google Fonts 加载（wght 600/700）
- 正文：系统字体栈（`-apple-system, "PingFang SC", "Microsoft YaHei"...`）
- 代码：`"SF Mono", "Cascadia Code", Consolas...`

**特效约定**：
- 用户气泡：便签效果（`transform: rotate(-0.5deg)`，hover 回正）
- AI 气泡：羊皮纸底色 + 细边框
- 代码块：深靛蓝渐变背景（`linear-gradient(135deg, #1a1a2e, var(--ink))`）
- 消息入场：`@keyframes msgFadeIn`（opacity 0→1 + translateY 8px→0）
- 打字光标：琥珀色竖线闪烁（`@keyframes blink`）
- 阴影：暖色调（`rgba(120, 90, 50, ...)`），不用冷灰

**新增页面/组件时**：
- 颜色只用 token（`var(--amber)` 而非 `#d97706`）
- Bootstrap 覆写已做（`.btn-primary`、`.nav-link.active`、`.navbar-dark`），直接用 Bootstrap 类
- 圆角统一 `var(--radius)`（12px），大卡片 16px
- 过渡统一 `var(--duration)`（200ms）+ `var(--ease)`
- 支持 `prefers-reduced-motion`（已全局禁用动画）
- 响应式断点：576px（气泡宽度）、768px（侧边栏折叠）

### 终端环境

默认用 bash 原生工具链（grep / sed / awk / find / curl）。仅当需要 Windows 原生能力（注册表/服务/WMI）时才从 bash 调 PowerShell。

---

## 2. 代码风格

### Python

- Python 3.12+，用较新语法（`X | None` 替代 `Optional[X]`，`list[str]` 替代 `List[str]`）
- 异步优先：IO 操作用 `async/await`，同步模型推理用 `asyncio.to_thread`
- 遵循向下规则：公开方法在文件顶部，细节逐层下沉
- 不要为简单线性逻辑拆出一堆细碎 helper
- 过度防御 = 掩盖设计缺陷，好的软件在预设条件下运行，其余情况报错修复

### JavaScript

- ES6+，模块化 IIFE 封装
- DOM 操作用原生 API，不引入框架
- SSE 帧解析器（`createSSEParser`）是自定义实现，不依赖 EventSource

---

## 3. 架构速查

```
用户 → FastAPI → stream_graph（手动编排）
  ├─ tool_router（显式联网命令 / doc_meta 元信息）
  ├─ intent_router（是否需要检索）
  ├─ rewrite_query（指代消解）
  ├─ retrieve_documents（dense+sparse RRF → reranker）
  ├─ grade_documents（score ≥ 0.5 → RAG / < 0.5 → CRAG 重查）
  ├─ transform_query（CRAG 变体重查）
  └─ generate_answer / general_answer / fallback / web_search
```

**关键配置**（`app/config.py`）：
- `retrieval_score_threshold=0.5`：rerank 分 ≥ 0.5 走严格 RAG，< 0.5 走 fallback
- `crag_max_attempts=1`：低相关时最多重查 1 次
- `dense_recall_top_k=50` → `retrieve_top_k=20` → `rerank_top_k=3`（漏斗型三层压缩）

---

## 4. 开发准则

### 改动前先想清楚

- 不确定就问，不猜
- 多方案时列取舍，不默默选一个
- 最小改动解决问题，不加没要求的功能

### 精准手术

- 只改必须改的，不顺手"优化"相邻代码
- 匹配已有代码风格，即使你觉得可以更好
- 你改出来的孤儿（import/变量/函数），你清掉

### SSE 流式注意事项

- 生成中（`streaming=true`）：发送按钮变红色停止按钮，可中断
- 停止后：`userStopped=true`，SSE handlers（token/reasoning/answer_final）检查此标志跳过 DOM 更新
- `CancelledError` 传播：`_finalize` 在 partial 时直接 `await`（非 shield），保证落库
- cursor 清理三层防线：`stopped` flag + catch 块 `querySelectorAll().remove()` + finally 块兜底

---

## 5. 测试

```bash
# 全量测试
.venv/Scripts/python -m pytest tests/ -v

# 单文件
.venv/Scripts/python -m pytest tests/test_continue.py -v

# 单个用例
.venv/Scripts/python -m pytest tests/test_chat.py::test_ask_empty_question -v
```

测试依赖 Redis（Docker 本地端口 6379）和 SQLite（自动创建 `data/test.db`）。

---

## 6. 文档知识库（项目笔记）

项目文档在 `D:\note\ck\project\个人项目\文档知识库问答\` 下：
- `DocQA可能会被问的问题.md` — 面试 Q&A 速查
- `项目中的困难与决策.md` — 踩坑记录与决策
- `流程图.md` — 架构流程图

改动影响面试回答时，同步更新对应文档。
