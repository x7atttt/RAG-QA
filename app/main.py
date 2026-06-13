import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, chat, documents
from app.core.database import init_db
from app.core.exceptions import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    await init_db()
    from app.agent.graph import build_graph

    app.state.graph = build_graph()
    yield


app = FastAPI(title="AI 智能文档问答系统", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
app.include_router(documents.router, prefix="/api/documents", tags=["文档"])
app.include_router(chat.router, prefix="/api/chat", tags=["对话"])


@app.get("/health")
async def health():
    return {"code": 0, "message": "ok", "data": {"status": "up"}}
