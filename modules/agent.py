"""
OmniService agentic graph — LangGraph implementation.

Graph flow:
    [START] → planner → executor ─┬─→ web_search → synthesizer → [END]
                                   └──────────────→ synthesizer → [END]

Features:
    - Persistence via checkpointer  (InMemorySaver or SqliteSaver)
    - Durable execution             (@task-wrapped side-effect calls)
    - Streaming final answer        (stream_mode="messages", filter by synthesizer node)

Usage (sync, streaming):
    from modules.agent import build_graph, stream_answer

    graph = build_graph()
    for token in stream_answer(
        graph,
        model_id="CPB050JC-S-0-EV",
        problem_description="What parts does it need?",
        user_message="What parts does the CPB050JC need?",
        thread_id="s1",
    ):
        print(token, end="", flush=True)

Usage (single invoke):
    from modules.agent import build_graph, invoke_answer

    answer = invoke_answer(
        graph,
        model_id="CPB050JC-S-0-EV",
        problem_description="What parts does it need?",
        user_message="What parts does the CPB050JC need?",
        thread_id="s1",
    )
"""

from __future__ import annotations
from modules.template import (
    PLANNER_SYSTEM,
    PLANNER_USER,
    REPORT_SYSTEM,
    REPORT_USER,
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_CONTEXT,
    SYNTHESIZER_PREFERENCES_SECTION,
    SYNTHESIZER_WEB_SECTION,
    WEB_SEARCH_SYSTEM,
    format_prompt,
)
from modules.search import search_knowledge_base, web_search_tool
from langgraph.func import task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain_core.runnables import RunnableConfig

import json
import logging
import os
import uuid
from typing import TypedDict, Annotated, AsyncGenerator, Generator, Any
from operator import add

logger = logging.getLogger("omniservice.agent")

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:
    SqliteSaver = None  # type: ignore[assignment]

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except Exception:
    AsyncSqliteSaver = None  # type: ignore[assignment]

_SQLITE_CHECKPOINTER_CM: Any | None = None
_SQLITE_CHECKPOINTER: Any | None = None
_ASYNC_SQLITE_CHECKPOINTER_CM: Any | None = None
_ASYNC_SQLITE_CHECKPOINTER: Any | None = None
_ASYNC_GRAPH: Any | None = None


# ────────────────────────────────────────────────────────────────────
# LLM
# ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def _get_llm(model_name: str = DEFAULT_MODEL) -> ChatAnthropic:
    return ChatAnthropic(model_name=model_name)  # type: ignore[call-arg]


# ────────────────────────────────────────────────────────────────────
# State
# ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Graph state that flows through every node.

    ``messages`` uses an *add* reducer so each node can append without
    overwriting prior history — this is how LangGraph accumulates the
    multi-turn conversation across checkpoints.
    """
    messages: Annotated[list,
                        # conversation history (HumanMessage + AIMessage only)
                        add]
    model_id: str                           # equipment model extracted by the planner
    # concise restatement of the technician's problem
    problem_description: str
    user_id: str
    user_preferences: dict
    plan: dict                              # planner JSON output
    tool_results: dict                      # raw results from KB / external search
    # whether to also search the web (frontend toggle)
    web_search: bool


# ────────────────────────────────────────────────────────────────────
# Durable tasks — side-effect calls wrapped for safe replay
# ────────────────────────────────────────────────────────────────────

@task
def _kb_search_task(model_id: str, problem_description: str) -> dict:
    """Search Weaviate + Neo4j.  Wrapped in @task so it is NOT re-executed
    when a workflow resumes from a later checkpoint."""
    return search_knowledge_base(model_id, problem_description)


# ────────────────────────────────────────────────────────────────────
# Graph nodes
# ────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """Analyse the user's message and produce a structured plan (JSON).

    ``model_id`` and ``problem_description`` are already provided by the
    frontend and live in state — the planner only decides *which tools*
    to call.
    """
    llm = _get_llm()

    model_id = state.get("model_id", "")
    problem_description = state.get("problem_description", "")

    system = format_prompt(
        PLANNER_SYSTEM,
        model_id=model_id,
        problem_description=problem_description,
    )
    user_msg = state["messages"][-1]  # latest HumanMessage
    user_text = user_msg.content if hasattr(
        user_msg, "content") else str(user_msg)

    messages = [
        SystemMessage(content=system),
        # include conversation history for context
        *state["messages"],
        HumanMessage(content=format_prompt(
            PLANNER_USER, user_message=user_text)),
    ]

    response = llm.invoke(
        messages,
        config=RunnableConfig(
            run_name="planner_llm",
            tags=["planner", "omniservice"],
            metadata={"model_id": model_id, "node": "planner"},
        ),
    )

    # Parse the planner JSON — gracefully handle LLM formatting quirks
    content = response.content
    raw = content if isinstance(content, str) else str(content)
    raw = raw.strip()
    # Strip optional markdown fences the LLM might add despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[: raw.rfind("```")]
    raw = raw.strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: assume KB search with whatever info we have
        plan = {
            "reasoning": raw,
            "tools_to_call": ["search_knowledge_base"],
        }

    return {
        "plan": plan,
        "messages": [],  # planner internals are NOT added to conversation
    }


def executor_node(state: AgentState) -> dict:
    """Execute the knowledge-base search (always runs)."""
    from langsmith import traceable  # type: ignore[import-untyped]

    tool_results: dict = {}

    model_id = state.get("model_id", "")
    problem = state.get("problem_description", "")

    @traceable(name="kb_search", tags=["executor", "omniservice"],
               metadata={"model_id": model_id})
    def _run_kb_search(mid: str, prob: str) -> dict:
        future = _kb_search_task(mid, prob)
        return future.result()

    if model_id and problem:
        tool_results = _run_kb_search(model_id, problem)

    return {
        "tool_results": tool_results,
        "messages": [],
    }


def web_search_node(state: AgentState) -> dict:
    """Let the LLM craft a web search query via tool-calling.

    The LLM receives the conversation context and KB results, then
    calls ``web_search_tool`` with a query it designs itself.
    Results are merged into ``tool_results["web_results"]``.
    """
    llm = _get_llm()
    llm_with_tool = llm.bind_tools([web_search_tool])

    user_msg = state["messages"][-1]
    user_text = user_msg.content if hasattr(
        user_msg, "content") else str(user_msg)

    system = format_prompt(
        WEB_SEARCH_SYSTEM,
        model_id=state.get("model_id", ""),
        problem_description=state.get("problem_description", ""),
        user_message=user_text,
    )

    messages = [
        SystemMessage(content=system),
        *state["messages"],
    ]

    response = llm_with_tool.invoke(
        messages,
        config=RunnableConfig(
            run_name="web_search_llm",
            tags=["web_search", "omniservice"],
            metadata={
                "model_id": state.get("model_id", ""),
                "node": "web_search",
            },
        ),
    )

    # Execute the tool call(s) the LLM made
    tool_results = dict(state.get("tool_results", {}))

    for tool_call in response.tool_calls:
        if tool_call["name"] == "web_search_tool":
            query = tool_call["args"].get("query", "")
            logger.info("LLM web search query: %s", query)
            result = web_search_tool.invoke(
                query,
                config=RunnableConfig(
                    run_name="duckduckgo_search",
                    tags=["web_search", "tool", "omniservice"],
                    metadata={"query": query},
                ),
            )
            logger.info("Web search returned %d chars", len(str(result)))
            tool_results["web_results"] = result

    return {
        "tool_results": tool_results,
        "messages": [],
    }


def synthesizer_node(state: AgentState) -> dict:
    """Generate the final technician-facing answer.

    This is the node whose LLM tokens are streamed to the user via
    ``stream_mode="messages"`` (filter by ``langgraph_node == "synthesizer"``).
    """
    llm = _get_llm()

    system = format_prompt(SYNTHESIZER_SYSTEM)

    # Build context block from tool results
    tr = state.get("tool_results", {})
    user_preferences = state.get("user_preferences", {})
    vector_results = json.dumps(
        tr.get("vector_results", {}), indent=2, default=str,
    )
    graph_context = json.dumps(
        tr.get("graph_context", {}), indent=2, default=str,
    )

    # Include web results section only when present
    web_results_raw = tr.get("web_results")
    if web_results_raw:
        web_section = format_prompt(
            SYNTHESIZER_WEB_SECTION,
            web_results=web_results_raw if isinstance(web_results_raw, str)
            else json.dumps(web_results_raw, indent=2, default=str),
        )
    else:
        web_section = ""

    if user_preferences:
        preferences_section = format_prompt(
            SYNTHESIZER_PREFERENCES_SECTION,
            user_preferences=json.dumps(
                user_preferences, indent=2, default=str),
        )
    else:
        preferences_section = ""

    context_msg = format_prompt(
        SYNTHESIZER_CONTEXT,
        model_id=state.get("model_id", "unknown"),
        preferences_section=preferences_section,
        vector_results=vector_results,
        graph_context=graph_context,
        web_section=web_section,
    )

    messages = [
        SystemMessage(content=system),
        # full conversation history
        *state["messages"],
        HumanMessage(content=context_msg),           # injected context
    ]

    # Even though we call .invoke(), LangGraph will still surface per-token
    # chunks when the caller uses stream_mode="messages".
    response = llm.invoke(
        messages,
        config=RunnableConfig(
            run_name="synthesizer_llm",
            tags=["synthesizer", "omniservice"],
            metadata={
                "model_id": state.get("model_id", "unknown"),
                "node": "synthesizer",
                "has_web_results": bool(web_results_raw),
            },
        ),
    )

    return {
        "messages": [AIMessage(content=response.content)],
    }


# ────────────────────────────────────────────────────────────────────
# Graph builder
# ────────────────────────────────────────────────────────────────────

def _route_after_executor(state: AgentState) -> str:
    """Conditional edge: go to web_search_node if user toggled web search,
    otherwise skip straight to synthesizer."""
    if state.get("web_search", False):
        return "web_search"
    return "synthesizer"


def build_graph(checkpointer=None):
    """Construct and compile the OmniService agent graph.

    Graph flow:
        [START] → planner → executor ─┬─ (web_search=true)  → web_search → synthesizer → [END]
                                       └─ (web_search=false) → synthesizer → [END]

    Args:
        checkpointer: A LangGraph checkpointer instance.  Defaults to
            ``InMemorySaver()`` (state persists in RAM for the process
            lifetime — swap with ``SqliteSaver`` for on-disk durability).

    Returns:
        A compiled ``StateGraph`` ready for ``.invoke()`` / ``.stream()``.
    """
    global _SQLITE_CHECKPOINTER_CM, _SQLITE_CHECKPOINTER

    if checkpointer is None:
        checkpoint_path = os.getenv(
            "LANGGRAPH_CHECKPOINT_DB", ".data/checkpoints.db")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if SqliteSaver is not None:
            # from_conn_string returns a context manager; keep it open
            # for the process lifetime so the saver remains usable.
            if _SQLITE_CHECKPOINTER is None:
                _SQLITE_CHECKPOINTER_CM = SqliteSaver.from_conn_string(
                    checkpoint_path)
                _SQLITE_CHECKPOINTER = _SQLITE_CHECKPOINTER_CM.__enter__()
            checkpointer = _SQLITE_CHECKPOINTER
        else:
            logger.warning(
                "langgraph-checkpoint-sqlite not available; using InMemorySaver fallback."
            )
            checkpointer = InMemorySaver()

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Edges
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", _route_after_executor)
    builder.add_edge("web_search", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer)


async def get_async_graph(checkpointer=None):
    """Construct and cache an async-compiled graph for async streaming.

    Uses ``AsyncSqliteSaver`` when available so ``graph.astream(...)`` and
    related async graph methods run with an async checkpointer.
    """
    global _ASYNC_SQLITE_CHECKPOINTER_CM, _ASYNC_SQLITE_CHECKPOINTER, _ASYNC_GRAPH

    if _ASYNC_GRAPH is not None and checkpointer is None:
        return _ASYNC_GRAPH

    if checkpointer is None:
        checkpoint_path = os.getenv(
            "LANGGRAPH_CHECKPOINT_DB", ".data/checkpoints.db")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if AsyncSqliteSaver is not None:
            # from_conn_string returns an async context manager; keep it open
            # for process lifetime so the saver remains usable.
            if _ASYNC_SQLITE_CHECKPOINTER is None:
                _ASYNC_SQLITE_CHECKPOINTER_CM = AsyncSqliteSaver.from_conn_string(
                    checkpoint_path,
                )
                _ASYNC_SQLITE_CHECKPOINTER = await _ASYNC_SQLITE_CHECKPOINTER_CM.__aenter__()
            checkpointer = _ASYNC_SQLITE_CHECKPOINTER
        else:
            logger.warning(
                "AsyncSqliteSaver not available; using sync graph/checkpointer fallback.",
            )
            return build_graph()

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", _route_after_executor)
    builder.add_edge("web_search", "synthesizer")
    builder.add_edge("synthesizer", END)
    compiled = builder.compile(checkpointer=checkpointer)
    if checkpointer is _ASYNC_SQLITE_CHECKPOINTER:
        _ASYNC_GRAPH = compiled
    return compiled


# ────────────────────────────────────────────────────────────────────
# Convenience helpers
# ────────────────────────────────────────────────────────────────────

def _make_config(
    thread_id: str | None = None,
    *,
    model_id: str = "",
    user_id: str = "",
    web_search: bool = False,
) -> dict:
    """Build a LangGraph config dict with a thread_id for persistence.

    Includes LangSmith ``run_name``, ``tags``, and ``metadata`` so the
    entire graph invocation is clearly labeled in the LangSmith dashboard.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    tags = ["omniservice"]
    if web_search:
        tags.append("web_search_enabled")

    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "OmniService Agent",
        "tags": tags,
        "metadata": {
            "thread_id": thread_id,
            "model_id": model_id,
            "user_id": user_id,
            "web_search": web_search,
        },
    }


def generate_chat_report(
    *,
    project_id: str,
    user_id: str,
    device_type: str,
    model_id: str,
    problem_description: str,
    start_time: str,
    end_time: str,
    messages: list[dict[str, str]],
) -> str:
    """Generate an end-of-chat markdown report from session context."""
    llm = _get_llm()

    conversation_lines: list[str] = []
    for msg in messages:
        role = (msg.get("role", "") or "unknown").strip().lower()
        content = (msg.get("content", "") or "").strip()
        if not content:
            continue
        speaker = "Technician" if role == "user" else "Assistant"
        conversation_lines.append(f"- {speaker}: {content}")

    conversation = "\n".join(
        conversation_lines) if conversation_lines else "- No conversation captured."

    system = format_prompt(REPORT_SYSTEM)
    user = format_prompt(
        REPORT_USER,
        project_id=project_id,
        user_id=user_id,
        device_type=device_type,
        model_id=model_id,
        problem_description=problem_description,
        start_time=start_time,
        end_time=end_time,
        conversation=conversation,
    )

    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ],
        config=RunnableConfig(
            run_name="report_llm",
            tags=["report", "omniservice"],
            metadata={
                "project_id": project_id,
                "user_id": user_id,
                "model_id": model_id,
            },
        ),
    )
    content = response.content
    return content if isinstance(content, str) else str(content)


def stream_answer(
    graph,
    *,
    model_id: str,
    problem_description: str,
    user_message: str,
    user_id: str = "",
    user_preferences: dict[str, Any] | None = None,
    web_search: bool = False,
    thread_id: str | None = None,
) -> Generator[str, None, None]:
    """Stream the synthesizer's answer token-by-token.

    Yields plain-text token strings as they arrive from the LLM.

    Args:
        graph:               A compiled agent graph (from ``build_graph``).
        model_id:            Equipment model ID provided by the frontend.
        problem_description: Problem description provided by the frontend.
        user_message:        The technician's free-text question.
        web_search:          Whether to also run a web search (frontend toggle).
        thread_id:           Session identifier for persistence.  If *None*
                             a new UUID is generated (fresh conversation).

    Yields:
        str: individual token strings from the synthesizer LLM.
    """
    config = _make_config(
        thread_id,
        model_id=model_id,
        user_id=user_id,
        web_search=web_search,
    )
    inputs = {
        "messages": [HumanMessage(content=user_message)],
        "model_id": model_id,
        "problem_description": problem_description,
        "user_id": user_id,
        "user_preferences": user_preferences or {},
        "web_search": web_search,
        "tool_results": {},
        "plan": {},
    }

    for message_chunk, metadata in graph.stream(
        inputs,
        config,
        stream_mode="messages",
    ):
        # Only yield incremental token chunks from the synthesizer node,
        # not the final complete AIMessage emitted when the node finishes.
        if (
            isinstance(message_chunk, AIMessageChunk)
            and message_chunk.content
            and metadata.get("langgraph_node") == "synthesizer"
        ):
            content = message_chunk.content
            yield content if isinstance(content, str) else str(content)


async def astream_answer(
    graph,
    *,
    model_id: str,
    problem_description: str,
    user_message: str,
    user_id: str = "",
    user_preferences: dict[str, Any] | None = None,
    web_search: bool = False,
    thread_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Async version of ``stream_answer`` — cancellable via asyncio.

    Use this in async frameworks (FastAPI) so that when the client
    disconnects (e.g. user clicks stop), the asyncio task is cancelled
    and the LLM stream is abandoned immediately.

    Yields:
        str: individual token strings from the synthesizer LLM.
    """
    config = _make_config(
        thread_id,
        model_id=model_id,
        user_id=user_id,
        web_search=web_search,
    )
    inputs = {
        "messages": [HumanMessage(content=user_message)],
        "model_id": model_id,
        "problem_description": problem_description,
        "user_id": user_id,
        "user_preferences": user_preferences or {},
        "web_search": web_search,
        "tool_results": {},
        "plan": {},
    }

    async for message_chunk, metadata in graph.astream(
        inputs,
        config,
        stream_mode="messages",
    ):
        if (
            isinstance(message_chunk, AIMessageChunk)
            and message_chunk.content
            and metadata.get("langgraph_node") == "synthesizer"
        ):
            content = message_chunk.content
            yield content if isinstance(content, str) else str(content)


def invoke_answer(
    graph,
    *,
    model_id: str,
    problem_description: str,
    user_message: str,
    user_id: str = "",
    user_preferences: dict[str, Any] | None = None,
    web_search: bool = False,
    thread_id: str | None = None,
) -> str:
    """Run the full graph and return the final answer as a single string.

    Args:
        graph:               A compiled agent graph (from ``build_graph``).
        model_id:            Equipment model ID provided by the frontend.
        problem_description: Problem description provided by the frontend.
        user_message:        The technician's free-text question.
        web_search:          Whether to also run a web search (frontend toggle).
        thread_id:           Session identifier for persistence.

    Returns:
        The synthesizer's complete answer text.
    """
    config = _make_config(
        thread_id,
        model_id=model_id,
        user_id=user_id,
        web_search=web_search,
    )
    inputs = {
        "messages": [HumanMessage(content=user_message)],
        "model_id": model_id,
        "problem_description": problem_description,
        "user_id": user_id,
        "user_preferences": user_preferences or {},
        "web_search": web_search,
        "tool_results": {},
        "plan": {},
    }

    result = graph.invoke(inputs, config)
    # The last AIMessage in messages is the synthesizer's answer
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""
