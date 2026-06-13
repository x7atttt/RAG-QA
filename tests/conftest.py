import asyncio
import os
from collections.abc import AsyncIterator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./data/test.db"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["CHROMA_PERSIST_DIR"] = "./data/chroma_test"


@pytest_asyncio.fixture(scope="function")
async def client() -> AsyncIterator[AsyncClient]:
    from app.core.database import Base, engine, init_db
    from app.main import app

    os.makedirs("data", exist_ok=True)

    await init_db()

    from app.agent.graph import build_graph

    if not getattr(app.state, "graph", None):
        app.state.graph = build_graph()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    import shutil

    if os.path.exists("data/chroma_test"):
        shutil.rmtree("data/chroma_test", ignore_errors=True)
