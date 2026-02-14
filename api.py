"""
OmniService API — FastAPI server with a single streaming chat endpoint.

Run:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modules.agent import build_graph, astream_answer

# ── App & graph ──────────────────────────────────────────────────────

app = FastAPI(title="OmniService AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
graph = build_graph()


# ── Request / response schemas ───────────────────────────────────────

class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    project_id: str
    device_type: str
    model_number: str
    problem_description: str
    messages: list[Message]


# ── Endpoint ─────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    """Stream a technician-facing answer token-by-token via SSE.

    The last ``user`` message in ``req.messages`` is treated as the
    current question.  ``project_id`` is used as the LangGraph
    ``thread_id`` so conversation state persists across requests.

    If the client disconnects (e.g. user clicks stop), the async
    generator is cancelled and the LLM stream is abandoned.
    """
    # Extract the latest user message
    user_message = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    async def event_stream():
        try:
            async for token in astream_answer(
                graph,
                model_id=req.model_number,
                problem_description=req.problem_description,
                user_message=user_message,
                thread_id=req.project_id,
            ):
                # Stop generating if the client disconnected
                if await request.is_disconnected():
                    break
                yield f"{token}"
            yield "[DONE]"
        except Exception:
            # Client abort / cancellation — exit cleanly
            return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
