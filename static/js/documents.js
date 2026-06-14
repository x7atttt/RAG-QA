/**
 * 文档管理页：上传（拖拽+点击）、游标分页列表、删除
 */
(function () {
    "use strict";
    if (!window.API.requireAuth()) return;

    const { Token, ENDPOINTS, fetchJSON, formatSize, formatTime } = window.API;

    // 顶部用户名
    const user = Token.getUser();
    if (user) document.getElementById("navUser").textContent = `👤 ${user.username}`;

    document.getElementById("logoutBtn").addEventListener("click", () => {
        Token.clear();
        location.href = "login.html";
    });

    // ---------- Toast ----------
    const toastEl = document.getElementById("toast");
    const toast = new bootstrap.Toast(toastEl, { delay: 2500 });
    function showToast(msg, type = "primary") {
        toastEl.className = `toast align-items-center text-bg-${type} border-0`;
        document.getElementById("toastBody").textContent = msg;
        toast.show();
    }

    // ---------- 列表 + 游标分页 ----------
    const tbody = document.getElementById("docTbody");
    const loadMoreBtn = document.getElementById("loadMoreBtn");
    const listMeta = document.getElementById("listMeta");
    let nextCursor = null;
    let hasMore = false;

    function rowHtml(doc) {
        return `<tr data-id="${doc.id}">
            <td><i class="bi ${fileIcon(doc.file_type)} me-2 text-muted"></i>${escapeHtml(doc.filename)}</td>
            <td><span class="badge bg-secondary">${escapeHtml(doc.file_type || "-")}</span></td>
            <td class="text-end">${doc.chunk_count ?? 0}</td>
            <td class="text-end">${formatSize(doc.file_size)}</td>
            <td class="small text-muted">${formatTime(doc.created_at)}</td>
            <td class="text-end">
                <button class="btn btn-sm btn-outline-danger del-btn"><i class="bi bi-trash"></i></button>
            </td>
        </tr>`;
    }

    function fileIcon(type) {
        if (type === "pdf") return "bi-file-earmark-pdf";
        if (type === "docx") return "bi-file-earmark-word";
        if (type === "md") return "bi-file-earmark-text";
        return "bi-file-earmark";
    }

    function escapeHtml(s) {
        return window.API.escapeHtml(s || "");
    }

    async function loadFirst() {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted py-4">加载中...</td></tr>`;
        nextCursor = null;
        hasMore = false;
        await loadMore();
    }

    async function loadMore() {
        loadMoreBtn.classList.add("d-none");
        try {
            const params = new URLSearchParams({ limit: "20" });
            if (nextCursor) params.set("cursor", nextCursor);
            const data = await fetchJSON(`${ENDPOINTS.documents.list}?${params}`);
            const docs = data.documents || [];
            if (!nextCursor && docs.length === 0) {
                tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted py-5">
                    <i class="bi bi-inbox fs-1 d-block mb-2"></i>还没有文档，上传一个试试吧
                </td></tr>`;
            } else {
                tbody.insertAdjacentHTML("beforeend", docs.map(rowHtml).join(""));
            }
            nextCursor = data.next_cursor;
            hasMore = data.has_next;
            listMeta.textContent = docs.length ? `共加载 ${tbody.children.length} 条` : "";
            if (hasMore) loadMoreBtn.classList.remove("d-none");
        } catch (err) {
            tbody.innerHTML = `<tr><td colspan="6" class="text-center text-danger py-4">${escapeHtml(err.message)}</td></tr>`;
        }
    }

    loadFirst();
    loadMoreBtn.addEventListener("click", loadMore);

    // ---------- 删除 ----------
    tbody.addEventListener("click", async (e) => {
        const btn = e.target.closest(".del-btn");
        if (!btn) return;
        const tr = btn.closest("tr");
        const id = tr.dataset.id;
        if (!confirm("确定删除该文档？相关向量也会一并清除。")) return;
        btn.disabled = true;
        try {
            await fetchJSON(ENDPOINTS.documents.delete(id), { method: "DELETE" });
            tr.remove();
            if (!tbody.children.length) loadFirst();
            showToast("删除成功", "success");
        } catch (err) {
            showToast(err.message, "danger");
            btn.disabled = false;
        }
    });

    // ---------- 上传（拖拽 + 点击）----------
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const uploadBox = document.getElementById("uploadProgress");
    const uploadFileName = document.getElementById("uploadFileName");
    const uploadStatus = document.getElementById("uploadStatus");

    dropZone.addEventListener("click", () => fileInput.click());
    ["dragover", "dragenter"].forEach((ev) =>
        dropZone.addEventListener(ev, (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        })
    );
    ["dragleave", "drop"].forEach((ev) =>
        dropZone.addEventListener(ev, (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        })
    );
    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files[0]) uploadFile(files[0]);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) uploadFile(fileInput.files[0]);
        fileInput.value = ""; // 允许重复选同一文件
    });

    async function uploadFile(file) {
        // 前端预校验
        const ext = file.name.split(".").pop().toLowerCase();
        if (!["pdf", "docx", "md"].includes(ext)) {
            showToast("仅支持 PDF / DOCX / MD", "danger");
            return;
        }
        if (file.size > 20 * 1024 * 1024) {
            showToast("文件超过 20MB 限制", "danger");
            return;
        }

        uploadFileName.textContent = file.name;
        uploadStatus.textContent = "上传与解析中...";
        uploadBox.classList.remove("d-none");

        const fd = new FormData();
        fd.append("file", file);
        const token = Token.get();

        try {
            const resp = await fetch(ENDPOINTS.documents.upload, {
                method: "POST",
                headers: token ? { Authorization: `Bearer ${token}` } : {},
                body: fd,
            });
            if (resp.status === 401) {
                Token.clear();
                location.href = "login.html";
                return;
            }
            const payload = await resp.json().catch(() => null);
            if (!payload || payload.code !== 0) {
                throw new Error(payload?.message || `上传失败 (HTTP ${resp.status})`);
            }
            uploadStatus.textContent = "完成";
            showToast("上传成功", "success");
            await loadFirst();
        } catch (err) {
            uploadStatus.textContent = "失败";
            showToast(err.message, "danger");
        } finally {
            setTimeout(() => uploadBox.classList.add("d-none"), 1500);
        }
    }
})();
