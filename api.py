"""
OmniService API — FastAPI server with a single streaming chat endpoint.

Run:
    uvicorn api:app --reload --port 8000

LangSmith tracing:
    Set these in your .env to enable full pipeline tracing:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=lsv2_...
        LANGCHAIN_PROJECT=OmniService
"""

from __future__ import annotations

import os
import logging
import asyncio

from dotenv import load_dotenv

load_dotenv()

# Enable LangSmith tracing if API key is present
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "OmniService")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modules.agent import build_graph, astream_answer

logger = logging.getLogger("omniservice.api")


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
    web_search: bool = False    # user toggles "search the web" in the UI


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
                web_search=req.web_search,
                thread_id=req.project_id,
            ):
                # Stop generating if the client disconnected
                if await request.is_disconnected():
                    break
                yield f"{token}"
            yield "[DONE]"
        except asyncio.CancelledError:
            # Client disconnected — exit cleanly
            return
        except Exception as exc:
            logger.exception("Error during streaming: %s", exc)
            yield f"[ERROR] {exc}"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
