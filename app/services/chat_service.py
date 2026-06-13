import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.agent.state import AgentState


def sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_graph(
    graph: CompiledStateGraph,
    user_id: int,
    question: str,
) -> AsyncIterator[tuple[str, Any]]:
    """yield (event_name, payload). event ∈ token/sources/done/error."""

    initial: AgentState = {
        "user_id": user_id,
        "question": question,
        "should_retrieve": False,
        "retrieved_docs": [],
        "sources": [],
        "answer_tokens": [],
        "answer": "",
        "error": None,
        "meta": {},
    }

    answer_parts: list[str] = []
    sources_emitted = False

    try:
        async for evt in graph.astream_events(initial, version="v2"):
            kind = evt.get("event")

            if kind == "on_chain_end" and evt.get("name") == "retrieve_documents":
                output = evt.get("data", {}).get("output")
                if isinstance(output, dict) and output.get("sources") and not sources_emitted:
                    sources_emitted = True
                    yield ("sources", output["sources"])

            elif kind == "on_chat_model_stream":
                chunk = evt.get("data", {}).get("chunk")
                if chunk is None:
                    continue
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    answer_parts.append(content)
                    yield ("token", content)

        answer = "".join(answer_parts)
        yield ("answer_final", answer)
        yield ("done", {"status": "ok"})
    except Exception as e:
        yield ("error", {"message": str(e)})
