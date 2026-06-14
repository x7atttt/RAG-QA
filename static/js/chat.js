/**
 * 对话页：SSE 流式问答 + 来源引用渲染
 *
 * 后端 SSE 协议（标准 event:/data: 帧）：
 *   event: sources       data: [SourceItem]        JSON，检索到的来源（token 之前最多发一次）
 *   event: token         data: <原始字符串>          裸字符串，LLM 增量内容
 *   event: answer_final  data: <JSON 字符串>        完整答案
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

    let streaming = false; // 是否正在接收流（防止并发）

    // ---------- DOM 渲染辅助 ----------
    function appendUserMsg(text) {
        const el = document.createElement("div");
        el.className = "msg-row user-row";
        el.innerHTML = `<div class="bubble user-bubble">${escapeHtml(text)}</div>`;
        chatBox.appendChild(el);
        scrollToBottom();
    }

    /** 创建一条助手消息，返回 { row, contentEl } 用于流式追加 */
    function createAssistantMsg() {
        const row = document.createElement("div");
        row.className = "msg-row assistant-row";
        row.innerHTML = `
            <div class="bubble assistant-bubble">
                <div class="assistant-content"><span class="typing-cursor"></span></div>
            </div>
            <div class="sources-area mt-2"></div>
        `;
        chatBox.appendChild(row);
        const contentEl = row.querySelector(".assistant-content");
        return { row, contentEl };
    }

    function renderSources(areaEl, sources) {
        if (!sources || !sources.length) return;
        const cards = sources
            .map((s, i) => {
                const name = escapeHtml(s.filename || "来源");
                const score = typeof s.score === "number" ? (s.score * 100).toFixed(0) + "%" : "";
                const snippet = escapeHtml((s.content || "").slice(0, 120)) + (s.content && s.content.length > 120 ? "…" : "");
                return `<div class="source-card">
                    <div class="d-flex justify-content-between">
                        <span class="fw-semibold"><i class="bi bi-link-45deg me-1"></i>${i + 1}. ${name}</span>
                        ${score ? `<span class="badge bg-success-subtle text-success">${score}</span>` : ""}
                    </div>
                    <div class="source-snippet small text-muted mt-1">${snippet}</div>
                </div>`;
            })
            .join("");
        areaEl.innerHTML = `<div class="sources-label small text-muted mb-1"><i class="bi bi-quote me-1"></i>参考来源</div>${cards}`;
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // ---------- SSE 帧解析器（跨 chunk 缓冲）----------
    // 后端按 "event: xxx\ndata: yyy\n\n" 推送，需要缓冲不完整帧
    function createSSEParser(handlers) {
        let buffer = "";
        return {
            feed(chunk) {
                buffer += chunk;
                // 按空行切分（标准 SSE 帧分隔）
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
        let event = "message"; // SSE 默认事件名
        const dataLines = [];
        frame.split("\n").forEach((line) => {
            if (line.startsWith("event:")) {
                event = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
                dataLines.push(line.slice(5).replace(/^ /, "")); // 去掉单个前导空格
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
        statusHint.textContent = "正在检索文档...";

        appendUserMsg(question);

        const { row, contentEl } = createAssistantMsg();
        const sourcesArea = row.querySelector(".sources-area");
        let fullAnswer = "";
        let cursor = contentEl.querySelector(".typing-cursor");

        function updateContent() {
            // 保留打字光标
            contentEl.innerHTML = renderMarkdown(fullAnswer) + `<span class="typing-cursor"></span>`;
            // 代码高亮
            contentEl.querySelectorAll("pre code").forEach((b) => {
                if (window.hljs) try { window.hljs.highlightElement(b); } catch {}
            });
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
            token: (raw) => {
                // token 的 data 是裸字符串，直接追加
                fullAnswer += raw;
                updateContent();
                if (statusHint.textContent === "正在检索文档...") statusHint.textContent = "正在生成回答...";
            },
            answer_final: (raw) => {
                // data 是 JSON 字符串，内容为完整答案
                try {
                    const obj = JSON.parse(raw);
                    if (typeof obj === "string") fullAnswer = obj;
                    else if (obj && typeof obj.answer === "string") fullAnswer = obj.answer;
                } catch {
                    // 退化为直接用 raw
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
            const resp = await fetch(ENDPOINTS.chat.ask, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                },
                body: JSON.stringify({ question }),
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
            cursor = contentEl.querySelector(".typing-cursor");
            if (cursor) cursor.remove();
        } catch (err) {
            contentEl.innerHTML = `<div class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i>${escapeHtml(err.message)}</div>`;
            statusHint.textContent = "出错";
        } finally {
            streaming = false;
            setSending(false);
            if (statusHint.textContent === "正在检索文档..." || statusHint.textContent === "正在生成回答...") {
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

    // Enter 发送，Shift+Enter 换行
    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            askForm.requestSubmit();
        }
    });

    // 文本框自适应高度
    function autoResize() {
        questionInput.style.height = "auto";
        questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
    }
    questionInput.addEventListener("input", autoResize);

    // 示例问题点击
    document.addEventListener("click", (e) => {
        if (e.target.classList.contains("example-q")) {
            e.preventDefault();
            questionInput.value = e.target.textContent;
            autoResize();
            questionInput.focus();
        }
    });
})();
