/**
 * 对话页：SSE 流式问答 + 来源引用渲染 + 推理过程折叠
 *
 * 后端 SSE 协议（标准 event:/data: 帧）：
 *   event: sources       data: [SourceItem]        JSON，检索到的来源（token 之前最多发一次）
 *   event: reasoning     data: <原始字符串>          裸字符串，推理过程增量（DeepSeek reasoner 等）
 *   event: token         data: <原始字符串>          裸字符串，LLM 正式答案增量
 *   event: answer_final  data: {"answer":...[, "reasoning":...]}  完整答案/推理
 *   event: done          data: {"status":"ok"[,"cache":"hit"|"wait"]}
 *   event: error         data: {"message":"..."}
 */
(function () {
    "use strict";
    if (!window.API.requireAuth()) return;

    const { Token, ENDPOINTS, renderMarkdown, escapeHtml } = window.API;

    // 用户名
    const user = Token.getUser();
    if (user) document.getElementById("navUser").textContent = `👤 ${user.username}`;

    document.getElementById("logoutBtn").addEventListener("click", () => {
        Token.clear();
        location.href = "/login.html";
    });

    const chatBox = document.getElementById("chatBox");
    const askForm = document.getElementById("askForm");
    const questionInput = document.getElementById("questionInput");
    const sendBtn = document.getElementById("sendBtn");
    const statusHint = document.getElementById("statusHint");
    const thinkingToggle = document.getElementById("thinkingToggle");

    // 深度思考开关：用 localStorage 记住用户偏好
    const THINKING_KEY = "docqa_thinking";
    thinkingToggle.checked = localStorage.getItem(THINKING_KEY) === "1";
    thinkingToggle.addEventListener("change", () => {
        localStorage.setItem(THINKING_KEY, thinkingToggle.checked ? "1" : "0");
    });

    let streaming = false; // 是否正在接收流（防止并发）

    // ---------- DOM 渲染辅助 ----------
    function appendUserMsg(text) {
        const el = document.createElement("div");
        el.className = "msg-row user-row";
        el.innerHTML = `<div class="bubble user-bubble">${escapeHtml(text)}</div>`;
        chatBox.appendChild(el);
        scrollToBottom();
    }

    /**
     * 创建一条助手消息，返回：
     *   { row, contentEl, reasoningBox, reasoningEl, sourcesArea }
     * - sourcesArea 在气泡内顶部（参考来源 chip 标注）
     * - reasoningBox 推理折叠面板，初始隐藏
     */
    function createAssistantMsg() {
        const row = document.createElement("div");
        row.className = "msg-row assistant-row";
        row.innerHTML = `
            <div class="bubble assistant-bubble">
                <div class="sources-area mb-2"></div>
                <details class="reasoning-panel mb-2" style="display:none">
                    <summary class="reasoning-summary">
                        <i class="bi bi-lightbulb me-1"></i>推理过程
                        <span class="reasoning-hint small text-muted ms-1">点击展开/收起</span>
                    </summary>
                    <div class="reasoning-content mt-1"></div>
                </details>
                <div class="assistant-content"><span class="typing-cursor"></span></div>
            </div>
        `;
        chatBox.appendChild(row);
        const contentEl = row.querySelector(".assistant-content");
        const reasoningBox = row.querySelector(".reasoning-panel");
        const reasoningEl = row.querySelector(".reasoning-content");
        const sourcesArea = row.querySelector(".sources-area");
        return { row, contentEl, reasoningBox, reasoningEl, sourcesArea };
    }

    function renderMarkdownHighlight(html) {
        // 代码高亮
        html.querySelectorAll("pre code").forEach((b) => {
            if (window.hljs) try { window.hljs.highlightElement(b); } catch {}
        });
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // ---------- 来源标注（气泡内顶部 chip 标签，纯展示）----------
    function renderSources(areaEl, sources) {
        if (!sources || !sources.length) return;
        // 提取不重复的文件名（多个 chunk 可能来自同一文档）
        const names = [...new Set(sources.map((s) => s.filename).filter(Boolean))];
        if (!names.length) return;
        const chips = names
            .map((n) => `<span class="source-chip"><i class="bi bi-file-earmark-text me-1"></i>${escapeHtml(n)}</span>`)
            .join("");
        areaEl.innerHTML = `<span class="sources-label me-1"><i class="bi bi-quote"></i> 参考</span>${chips}`;
    }

    // ---------- SSE 帧解析器（跨 chunk 缓冲）----------
    function createSSEParser(handlers) {
        let buffer = "";
        return {
            feed(chunk) {
                buffer += chunk;
                let idx;
                while ((idx = buffer.indexOf("\n\n")) !== -1) {
                    const frame = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 2);
                    parseFrame(frame, handlers);
                }
            },
            flush() {
                if (buffer.trim()) parseFrame(buffer, handlers);
                buffer = "";
            },
        };
    }

    function parseFrame(frame, handlers) {
        let event = "message";
        const dataLines = [];
        frame.split("\n").forEach((line) => {
            if (line.startsWith("event:")) {
                event = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
                dataLines.push(line.slice(5).replace(/^ /, ""));
            }
        });
        if (!dataLines.length) return;
        const raw = dataLines.join("\n");
        const h = handlers[event] || handlers.message;
        if (h) h(raw);
    }

    // ---------- 发送问答 ----------
    async function ask(question) {
        if (streaming) return;
        streaming = true;
        setSending(true);
        const thinking = thinkingToggle.checked;
        statusHint.textContent = thinking ? "深度思考中..." : "正在检索文档...";

        appendUserMsg(question);

        const { contentEl, reasoningBox, reasoningEl, sourcesArea } = createAssistantMsg();
        let fullAnswer = "";
        let fullReasoning = "";

        function updateContent() {
            contentEl.innerHTML = renderMarkdown(fullAnswer) + `<span class="typing-cursor"></span>`;
            renderMarkdownHighlight(contentEl);
            scrollToBottom();
        }

        function updateReasoning() {
            // 推理内容用纯文本 + 换行保留（不渲染 markdown，避免与正文混淆）
            reasoningEl.innerHTML = `<pre class="reasoning-pre">${escapeHtml(fullReasoning)}</pre>`;
            reasoningBox.style.display = "block";
            scrollToBottom();
        }

        const parser = createSSEParser({
            sources: (raw) => {
                try {
                    const arr = JSON.parse(raw);
                    renderSources(sourcesArea, arr);
                } catch {}
                statusHint.textContent = "正在生成回答...";
            },
            reasoning: (raw) => {
                fullReasoning += raw;
                updateReasoning();
                if (statusHint.textContent === "正在检索文档...") statusHint.textContent = "正在推理...";
            },
            token: (raw) => {
                fullAnswer += raw;
                updateContent();
                if (statusHint.textContent === "正在检索文档..." || statusHint.textContent === "正在推理...") {
                    statusHint.textContent = "正在生成回答...";
                }
            },
            answer_final: (raw) => {
                try {
                    const obj = JSON.parse(raw);
                    if (obj && typeof obj.answer === "string") fullAnswer = obj.answer;
                    if (obj && typeof obj.reasoning === "string" && obj.reasoning) {
                        fullReasoning = obj.reasoning;
                        updateReasoning();
                    }
                } catch {
                    if (raw) fullAnswer = raw;
                }
                updateContent();
            },
            done: (raw) => {
                let cacheTag = "";
                try {
                    const obj = JSON.parse(raw);
                    if (obj.cache === "hit") cacheTag = " · 缓存命中";
                    else if (obj.cache === "wait") cacheTag = " · 缓存等待命中";
                } catch {}
                statusHint.textContent = `完成${cacheTag}`;
            },
            error: (raw) => {
                let msg = "回答失败";
                try { msg = JSON.parse(raw).message || msg; } catch { if (raw) msg = raw; }
                contentEl.innerHTML = `<div class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i>${escapeHtml(msg)}</div>`;
                statusHint.textContent = "出错";
            },
        });

        try {
            const token = Token.get();
            const thinking = thinkingToggle.checked;
            const resp = await fetch(ENDPOINTS.chat.ask, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                },
                body: JSON.stringify({ question, thinking }),
            });

            if (resp.status === 401) {
                Token.clear();
                location.href = "/login.html";
                return;
            }
            if (!resp.ok && !resp.headers.get("content-type")?.includes("text/event-stream")) {
                const payload = await resp.json().catch(() => null);
                throw new Error(payload?.message || `请求失败 (HTTP ${resp.status})`);
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                parser.feed(decoder.decode(value, { stream: true }));
            }
            parser.flush();

            // 流结束后移除打字光标
            const cursor = contentEl.querySelector(".typing-cursor");
            if (cursor) cursor.remove();
        } catch (err) {
            contentEl.innerHTML = `<div class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i>${escapeHtml(err.message)}</div>`;
            statusHint.textContent = "出错";
        } finally {
            streaming = false;
            setSending(false);
            if (statusHint.textContent === "正在检索文档..." || statusHint.textContent === "正在生成回答..." || statusHint.textContent === "正在推理...") {
                statusHint.textContent = "完成";
            }
        }
    }

    function setSending(sending) {
        sendBtn.disabled = sending;
        questionInput.disabled = sending;
        if (!sending) questionInput.focus();
    }

    // ---------- 事件绑定 ----------
    askForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const q = questionInput.value.trim();
        if (!q || streaming) return;
        questionInput.value = "";
        autoResize();
        ask(q);
    });

    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            askForm.requestSubmit();
        }
    });

    function autoResize() {
        questionInput.style.height = "auto";
        questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
    }
    questionInput.addEventListener("input", autoResize);

    document.addEventListener("click", (e) => {
        if (e.target.classList.contains("example-q")) {
            e.preventDefault();
            questionInput.value = e.target.textContent;
            autoResize();
            questionInput.focus();
        }
    });
})();
