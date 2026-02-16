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
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from modules.agent import astream_answer, build_graph, get_async_graph
from modules.persistence import (
    delete_project,
    get_project,
    list_projects_for_user,
    get_user_preferences,
    infer_and_update_user_preferences,
    init_state_db,
    touch_project,
)

load_dotenv()

# Enable LangSmith tracing if API key is present
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "OmniService")


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
init_state_db()
graph = build_graph()


# ── Request / response schemas ───────────────────────────────────────

class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    project_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
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

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in request.messages")

    # Project routing: create if missing, resume if existing.
    project_meta = touch_project(
        req.project_id,
        req.user_id,
        create_if_missing=True,
        allow_join_existing=False,
    )
    if project_meta.get("status") == "forbidden":
        raise HTTPException(
            status_code=403,
            detail=f"User '{req.user_id}' is not allowed to access project '{req.project_id}'",
        )
    user_preferences = get_user_preferences(req.user_id)
    logger.info(
        "Project %s (%s) for user %s",
        req.project_id,
        "created" if project_meta.get("created") else "resumed",
        req.user_id,
    )

    async def event_stream():
        full_answer = ""
        try:
            async for token in astream_answer(
                await get_async_graph(),
                model_id=req.model_number,
                problem_description=req.problem_description,
                user_message=user_message,
                user_id=req.user_id,
                user_preferences=user_preferences,
                web_search=req.web_search,
                thread_id=req.project_id,
            ):
                # Stop generating if the client disconnected
                if await request.is_disconnected():
                    break
                full_answer += token
                yield f"{token}"

            # Update inferred preferences after each complete answer.
            if full_answer:
                infer_and_update_user_preferences(
                    req.user_id, user_message=user_message, assistant_message=full_answer
                )
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


@app.get("/api/projects/{project_id}/history")
async def project_history(project_id: str, user_id: str = Query(..., min_length=1)):
    """
    Return project chat history so a technician can resume the same project.

    If the project exists, the user is associated to the project for handoff.
    """
    access = touch_project(
        project_id,
        user_id,
        create_if_missing=False,
        allow_join_existing=False,
    )
    status = access.get("status")
    if status == "not_found":
        return {
            "project_id": project_id,
            "exists": False,
            "messages": [],
            "model_id": "",
            "problem_description": "",
            "users": [],
        }
    if status == "forbidden":
        raise HTTPException(
            status_code=403,
            detail=f"User '{user_id}' is not allowed to view project '{project_id}'",
        )
    snapshot = graph.get_state({"configurable": {"thread_id": project_id}})
    values = snapshot.values if getattr(snapshot, "values", None) else {}
    messages = values.get("messages", [])

    serialized: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            serialized.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            serialized.append({"role": "assistant", "content": str(msg.content)})

    latest_project = get_project(project_id) or access
    return {
        "project_id": project_id,
        "exists": True,
        "model_id": values.get("model_id", ""),
        "problem_description": values.get("problem_description", ""),
        "messages": serialized,
        "users": [u["user_id"] for u in latest_project.get("users", [])],
    }


@app.get("/api/projects")
async def projects_for_user(user_id: str = Query(..., min_length=1)):
    """List only projects associated with the provided user_id."""
    return {
        "user_id": user_id,
        "projects": list_projects_for_user(user_id),
    }


@app.get("/api/users/{user_id}/preferences")
async def user_preferences(user_id: str):
    """Return inferred technician preferences for UI/debugging."""
    return {
        "user_id": user_id,
        "preferences": get_user_preferences(user_id),
    }


@app.delete("/api/projects/{project_id}")
async def project_delete(project_id: str, user_id: str = Query(..., min_length=1)):
    """Delete project metadata and persisted conversation for this project_id."""
    result = delete_project(project_id=project_id, user_id=user_id)
    status = result.get("status")

    if status == "not_found":
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    if status == "forbidden":
        raise HTTPException(
            status_code=403,
            detail=f"User '{user_id}' is not allowed to delete project '{project_id}'",
        )
    return result
