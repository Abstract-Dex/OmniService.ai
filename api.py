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
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from modules.agent import (
    astream_answer,
    build_graph,
    generate_chat_report,
    get_async_graph,
)
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
MANUALS_ROOT = os.getenv("MANUALS_ROOT", "/data/manuals")


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


class ReportRequest(BaseModel):
    project_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    device_type: str
    model_number: str
    problem_description: str
    start_time: str
    end_time: str
    messages: list[Message] = Field(default_factory=list)


# ── Endpoint ─────────────────────────────────────────────────────────
def _extract_references_from_state(values: dict) -> list[dict[str, str | int | float | None]]:
    """Extract compact PDF references from latest vector retrieval results."""
    tool_results = values.get(
        "tool_results", {}) if isinstance(values, dict) else {}
    vector_results = tool_results.get(
        "vector_results", {}) if isinstance(tool_results, dict) else {}
    ranked = vector_results.get("results", []) if isinstance(
        vector_results, dict) else []

    references: list[dict[str, str | int | float | None]] = []
    seen: set[tuple[str, int | None]] = set()
    for item in ranked:
        if not isinstance(item, dict):
            continue
        manual_name = str(item.get("manual_name", "") or "")
        page_number = item.get("page_number")
        page_value = page_number if isinstance(page_number, int) else None
        key = (manual_name, page_value)
        if key in seen:
            continue
        seen.add(key)

        rank_raw = item.get("rank")
        score_raw = item.get("rerank_score")
        score_value: float | None = None
        if isinstance(score_raw, (int, float)):
            score_value = float(score_raw)
        references.append(
            {
                "manual_name": manual_name,
                "manual_url": f"/api/manuals/{quote(manual_name)}" if manual_name else "",
                "page_number": page_value,
                "model_id": str(item.get("model_id", "") or ""),
                "rank": int(rank_raw) if isinstance(rank_raw, int) else None,
                "rerank_score": score_value,
            }
        )
    return references


@app.get("/api/manuals/{manual_name}")
async def get_manual(manual_name: str):
    """Serve PDF manuals from MANUALS_ROOT for frontend document viewer."""
    root = Path(MANUALS_ROOT).resolve()
    safe_name = Path(manual_name).name
    if safe_name != manual_name:
        raise HTTPException(status_code=400, detail="Invalid manual_name")
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF manuals are supported")

    manual_path = (root / safe_name).resolve()
    if root not in manual_path.parents:
        raise HTTPException(status_code=400, detail="Invalid manual path")
    if not manual_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Manual '{safe_name}' not found")

    return FileResponse(
        path=str(manual_path),
        media_type="application/pdf",
        filename=safe_name,
    )


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
        raise HTTPException(
            status_code=400, detail="No user message found in request.messages")

    # Project routing: create if missing, resume if existing.
    project_meta = touch_project(
        req.project_id,
        req.user_id,
        model_id=req.model_number,
        problem_description=req.problem_description,
        create_if_missing=True,
        # Handoff: if project exists and user is new, join project membership.
        allow_join_existing=True,
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


@app.post("/api/chat/report")
async def create_chat_report(req: ReportRequest):
    """Generate a non-streaming markdown report for a completed chat session."""
    access = touch_project(
        req.project_id,
        req.user_id,
        create_if_missing=False,
        allow_join_existing=False,
    )
    status = access.get("status")
    if status == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"Project '{req.project_id}' not found",
        )
    if status == "forbidden":
        raise HTTPException(
            status_code=403,
            detail=f"User '{req.user_id}' is not allowed to report on project '{req.project_id}'",
        )

    snapshot = graph.get_state({"configurable": {"thread_id": req.project_id}})
    values = snapshot.values if getattr(snapshot, "values", None) else {}
    persisted_messages = values.get("messages", [])
    serialized: list[dict[str, str]] = []
    for msg in persisted_messages:
        if isinstance(msg, HumanMessage):
            serialized.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            serialized.append(
                {"role": "assistant", "content": str(msg.content)})

    input_messages = (
        [{"role": m.role, "content": m.content} for m in req.messages]
        if req.messages
        else serialized
    )

    model_id = req.model_number or values.get("model_id", "")
    problem_description = req.problem_description or values.get(
        "problem_description", "")

    report_markdown = generate_chat_report(
        project_id=req.project_id,
        user_id=req.user_id,
        device_type=req.device_type,
        model_id=model_id,
        problem_description=problem_description,
        start_time=req.start_time,
        end_time=req.end_time,
        messages=input_messages,
    )

    return {
        "project_id": req.project_id,
        "user_id": req.user_id,
        "model_id": model_id,
        "report_markdown": report_markdown,
    }


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
        # Handoff: opening history can attach a new technician to the project.
        allow_join_existing=True,
    )
    status = access.get("status")
    if status == "not_found":
        return {
            "project_id": project_id,
            "exists": False,
            "messages": [],
            "references": [],
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
            serialized.append(
                {"role": "assistant", "content": str(msg.content)})

    latest_project = get_project(project_id) or access
    references = _extract_references_from_state(values)
    return {
        "project_id": project_id,
        "exists": True,
        "model_id": values.get("model_id", latest_project.get("model_id", "")),
        "problem_description": values.get(
            "problem_description", latest_project.get(
                "problem_description", "")
        ),
        "messages": serialized,
        "references": references,
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
        raise HTTPException(
            status_code=404, detail=f"Project '{project_id}' not found")
    if status == "forbidden":
        raise HTTPException(
            status_code=403,
            detail=f"User '{user_id}' is not allowed to delete project '{project_id}'",
        )
    return result
