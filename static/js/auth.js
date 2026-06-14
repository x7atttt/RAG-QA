/**
 * 登录/注册页逻辑
 */
(function () {
    "use strict";
    const { Token, ENDPOINTS } = window.API;

    // 已登录则直接跳对话页
    if (Token.exists()) {
        location.href = "/chat.html";
        return;
    }

    const alertBox = document.getElementById("authAlert");

    function showAlert(msg, type = "danger") {
        alertBox.className = `alert mt-3 mb-0 alert-${type}`;
        alertBox.textContent = msg;
        alertBox.classList.remove("d-none");
    }

    function hideAlert() {
        alertBox.classList.add("d-none");
    }

    function setBtnLoading(btn, loading, text) {
        if (loading) {
            btn.dataset.text = btn.textContent;
            btn.disabled = true;
            btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>${text || "处理中..."}`;
        } else {
            btn.disabled = false;
            btn.textContent = btn.dataset.text || btn.textContent;
        }
    }

    function formToObj(form) {
        const obj = {};
        new FormData(form).forEach((v, k) => (obj[k] = v));
        return obj;
    }

    /** 登录成功后跳转 */
    function onAuthSuccess(data) {
        Token.set(data.access_token, data.user);
        location.href = "/chat.html";
    }

    // ---------- 登录 ----------
    document.getElementById("loginForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        hideAlert();
        const btn = document.getElementById("loginBtn");
        const body = JSON.stringify(formToObj(e.target));
        setBtnLoading(btn, true);
        try {
            const data = await window.API.fetchJSON(ENDPOINTS.login, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body,
            });
            onAuthSuccess(data);
        } catch (err) {
            showAlert(err.message);
            setBtnLoading(btn, false);
        }
    });

    // ---------- 注册 ----------
    document.getElementById("registerForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        hideAlert();
        const form = e.target;
        const data = formToObj(form);
        if (data.password !== data.password2) {
            showAlert("两次输入的密码不一致");
            return;
        }
        const btn = document.getElementById("registerBtn");
        setBtnLoading(btn, true);
        try {
            // 注册成功后直接走登录，省去用户二次输入
            await window.API.fetchJSON(ENDPOINTS.register, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: data.username, password: data.password }),
            });
            const loginData = await window.API.fetchJSON(ENDPOINTS.login, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: data.username, password: data.password }),
            });
            onAuthSuccess(loginData);
        } catch (err) {
            showAlert(err.message);
            setBtnLoading(btn, false);
        }
    });

    // 切换 tab 时清空提示
    document.querySelectorAll('button[data-bs-toggle="pill"]').forEach((b) => {
        b.addEventListener("click", hideAlert);
    });
})();
