/**
 * 公共工具：token 管理、统一 fetch 封装、Markdown 渲染、页面守卫
 * 全局对象挂载在 window.API 上，各页面直接调用
 */
(function () {
    "use strict";

    const TOKEN_KEY = "jwt_token";
    const USER_KEY = "jwt_user";

    // ---------- token 管理 ----------
    const Token = {
        get: () => localStorage.getItem(TOKEN_KEY),
        set: (token, user) => {
            localStorage.setItem(TOKEN_KEY, token);
            if (user) localStorage.setItem(USER_KEY, JSON.stringify(user));
        },
        getUser: () => {
            const raw = localStorage.getItem(USER_KEY);
            try {
                return raw ? JSON.parse(raw) : null;
            } catch {
                return null;
            }
        },
        clear: () => {
            localStorage.removeItem(TOKEN_KEY);
            localStorage.removeItem(USER_KEY);
        },
        exists: () => !!localStorage.getItem(TOKEN_KEY),
    };

    // ---------- API 端点常量 ----------
    const ENDPOINTS = {
        register: "/api/auth/register",
        login: "/api/auth/login",
        documents: {
            upload: "/api/documents/upload",
            list: "/api/documents/list",
            delete: (id) => `/api/documents/${id}`,
        },
        chat: {
            ask: "/api/chat/ask",
            history: "/api/chat/history",
            conversations: "/api/chat/conversations",
            deleteConversation: (id) => `/api/chat/conversations/${id}`,
        },
    };

    // ---------- 业务错误码映射（部分）----------
    const ERROR_MESSAGES = {
        10001: "认证失败，请重新登录",
        10003: "Token 无效或已过期",
        10004: "用户名已存在",
        10005: "用户名或密码错误",
        20001: "文档不存在或无权访问",
        30002: "问题不能为空",
    };

    function errText(code, fallback) {
        return ERROR_MESSAGES[code] || fallback || "请求失败";
    }

    /**
     * 统一 JSON 请求：自动注入 Authorization，解析 {code,message,data}，401 拦截跳登录
     * @param {string} path 端点路径
     * @param {object} options fetch options（method/body 等）
     * @returns {Promise<any>} 成功时 resolve(data)，失败时 reject(Error)
     */
    async function fetchJSON(path, options = {}) {
        const token = Token.get();
        const headers = { ...(options.headers || {}) };
        if (token) headers["Authorization"] = `Bearer ${token}`;

        let resp;
        try {
            resp = await fetch(path, { ...options, headers });
        } catch (e) {
            throw new Error("网络错误，请检查服务是否启动");
        }

        // 401 → 清 token 跳登录
        if (resp.status === 401) {
            Token.clear();
            redirectLogin();
            throw new Error("登录已失效，请重新登录");
        }

        const payload = await resp.json().catch(() => null);
        if (!payload || typeof payload.code !== "number") {
            throw new Error(`响应格式异常 (HTTP ${resp.status})`);
        }
        if (payload.code !== 0) {
            throw new Error(errText(payload.code, payload.message));
        }
        return payload.data;
    }

    function redirectLogin() {
        // 同时兼容 /login.html 和 /static/login.html 两种访问路径
        if (!location.pathname.endsWith("login.html")) {
            location.href = "/login.html";
        }
    }

    /**
     * 页面守卫：无 token 直接跳登录
     */
    function requireAuth() {
        if (!Token.exists()) {
            redirectLogin();
            return false;
        }
        return true;
    }

    // ---------- Markdown 渲染（marked + DOMPurify，安全净化）----------
    let markedReady = typeof window.marked !== "undefined";
    let purifyReady = typeof window.DOMPurify !== "undefined";

    /**
     * 把 Markdown 文本渲染成安全 HTML
     * 依赖：marked.js、DOMPurify（页面需引入 CDN）
     */
    function renderMarkdown(text) {
        if (!text) return "";
        let html;
        if (markedReady) {
            try {
                // 同步解析；marked v5+ 推荐 .parse，v4 用 .marked
                const parse = window.marked.parse || window.marked;
                html = parse(text);
            } catch {
                html = escapeHtml(text);
            }
        } else {
            html = `<pre>${escapeHtml(text)}</pre>`;
        }
        if (purifyReady) {
            try {
                html = window.DOMPurify.sanitize(html);
            } catch {
                /* 净化失败则保留原样 */
            }
        }
        return html;
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    // ---------- 工具：字节大小格式化、时间格式化 ----------
    function formatSize(bytes) {
        if (bytes == null) return "-";
        const n = Number(bytes);
        if (n < 1024) return `${n} B`;
        if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
        return `${(n / 1024 / 1024).toFixed(2)} MB`;
    }

    function formatTime(iso) {
        if (!iso) return "-";
        const d = new Date(iso);
        if (isNaN(d.getTime())) return iso;
        const pad = (x) => String(x).padStart(2, "0");
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    }

    // ---------- 对外暴露 ----------
    window.API = {
        Token,
        ENDPOINTS,
        fetchJSON,
        requireAuth,
        redirectLogin,
        renderMarkdown,
        escapeHtml,
        formatSize,
        formatTime,
        errText,
    };
})();
