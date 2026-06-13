import io

import pytest

from app.services.document_service import get_user_collection


async def _register_and_login(client, username: str = "alice", password: str = "secret123"):
    await client.post("/api/auth/register", json={"username": username, "password": password})
    resp = await client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.json()["data"]["access_token"]


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_upload_markdown(client):
    token = await _register_and_login(client)
    content = "# 测试文档\n\n这是一个用于 RAG 测试的文档。" * 50
    resp = await client.post(
        "/api/documents/upload",
        headers=_auth_header(token),
        files={"file": ("test.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["filename"] == "test.md"
    assert body["data"]["file_type"] == "md"
    assert body["data"]["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_upload_unsupported_type(client):
    token = await _register_and_login(client)
    resp = await client.post(
        "/api/documents/upload",
        headers=_auth_header(token),
        files={"file": ("evil.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == 20004


@pytest.mark.asyncio
async def test_upload_unauthenticated(client):
    resp = await client.post(
        "/api/documents/upload",
        files={"file": ("t.md", io.BytesIO(b"hello"), "text/markdown")},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_list_and_delete(client):
    token = await _register_and_login(client)
    for i in range(3):
        content = f"# 文档 {i}\n\n" + "内容 " * 200
        await client.post(
            "/api/documents/upload",
            headers=_auth_header(token),
            files={"file": (f"doc{i}.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
        )

    resp = await client.get("/api/documents/list", headers=_auth_header(token))
    assert resp.status_code == 200
    docs = resp.json()["data"]["documents"]
    assert len(docs) == 3

    doc_id = docs[0]["id"]
    del_resp = await client.delete(f"/api/documents/{doc_id}", headers=_auth_header(token))
    assert del_resp.status_code == 200
    assert del_resp.json()["code"] == 0

    resp2 = await client.get("/api/documents/list", headers=_auth_header(token))
    assert len(resp2.json()["data"]["documents"]) == 2


@pytest.mark.asyncio
async def test_user_isolation_chroma(client):
    token_a = await _register_and_login(client, "userA", "secret123")
    token_b = await _register_and_login(client, "userB", "secret123")

    content = "# 隔离测试\n\n" + "独占内容 " * 100
    await client.post(
        "/api/documents/upload",
        headers=_auth_header(token_a),
        files={"file": ("a.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
    )

    coll_a = get_user_collection(1)
    coll_b = get_user_collection(2)
    assert coll_a.count() > 0
    assert coll_b.count() == 0


@pytest.mark.asyncio
async def test_cursor_pagination(client):
    token = await _register_and_login(client)
    for i in range(5):
        content = f"# 文档 {i}\n\n" + "x " * 200
        await client.post(
            "/api/documents/upload",
            headers=_auth_header(token),
            files={"file": (f"d{i}.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
        )

    resp = await client.get("/api/documents/list?limit=2", headers=_auth_header(token))
    data = resp.json()["data"]
    assert len(data["documents"]) == 2
    assert data["has_next"] is True
    cursor = data["next_cursor"]

    resp2 = await client.get(
        f"/api/documents/list?limit=2&cursor={cursor}", headers=_auth_header(token)
    )
    data2 = resp2.json()["data"]
    assert len(data2["documents"]) == 2
    assert data2["documents"][0]["id"] < data["documents"][-1]["id"]
