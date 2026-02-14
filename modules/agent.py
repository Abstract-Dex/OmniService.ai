"""
OmniService agentic graph — LangGraph implementation.

Graph flow:
    [START] → planner → executor → synthesizer → [END]

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

import json
import uuid
from typing import TypedDict, Annotated, AsyncGenerator, Generator
from operator import add

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task

from modules.search import search_knowledge_base
from modules.template import (
    PLANNER_SYSTEM,
    PLANNER_USER,
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_CONTEXT,
    format_prompt,
)


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
    plan: dict                              # planner JSON output
    tool_results: dict                      # raw results from KB / external search


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

    response = llm.invoke(messages)

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
    """Execute tool calls dictated by the planner."""
    plan = state.get("plan", {})
    tools = plan.get("tools_to_call", [])
    tool_results: dict = {}

    if "search_knowledge_base" in tools:
        model_id = state.get("model_id", "")
        problem = state.get("problem_description", "")
        if model_id and problem:
            future = _kb_search_task(model_id, problem)
            tool_results = future.result()

    # TODO: add external_search tool here when implemented

    return {
        "tool_results": tool_results,
        "messages": [],  # tool outputs are NOT added to conversation messages
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
    vector_results = json.dumps(
        tr.get("vector_results", {}), indent=2, default=str,
    )
    graph_context = json.dumps(
        tr.get("graph_context", {}), indent=2, default=str,
    )
    context_msg = format_prompt(
        SYNTHESIZER_CONTEXT,
        model_id=state.get("model_id", "unknown"),
        vector_results=vector_results,
        graph_context=graph_context,
    )

    messages = [
        SystemMessage(content=system),
        # full conversation history
        *state["messages"],
        HumanMessage(content=context_msg),           # injected context
    ]

    # Even though we call .invoke(), LangGraph will still surface per-token
    # chunks when the caller uses stream_mode="messages".
    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content)],
    }


# ────────────────────────────────────────────────────────────────────
# Graph builder
# ────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """Construct and compile the OmniService agent graph.

    Args:
        checkpointer: A LangGraph checkpointer instance.  Defaults to
            ``InMemorySaver()`` (state persists in RAM for the process
            lifetime — swap with ``SqliteSaver`` for on-disk durability).

    Returns:
        A compiled ``StateGraph`` ready for ``.invoke()`` / ``.stream()``.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Edges  (linear for now — add conditional replan loop later)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer)


# ────────────────────────────────────────────────────────────────────
# Convenience helpers
# ────────────────────────────────────────────────────────────────────

def _make_config(thread_id: str | None = None) -> dict:
    """Build a LangGraph config dict with a thread_id for persistence."""
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    return {"configurable": {"thread_id": thread_id}}


def stream_answer(
    graph,
    *,
    model_id: str,
    problem_description: str,
    user_message: str,
    thread_id: str | None = None,
) -> Generator[str, None, None]:
    """Stream the synthesizer's answer token-by-token.

    Yields plain-text token strings as they arrive from the LLM.

    Args:
        graph:               A compiled agent graph (from ``build_graph``).
        model_id:            Equipment model ID provided by the frontend.
        problem_description: Problem description provided by the frontend.
        user_message:        The technician's free-text question.
        thread_id:           Session identifier for persistence.  If *None*
                             a new UUID is generated (fresh conversation).

    Yields:
        str: individual token strings from the synthesizer LLM.
    """
    config = _make_config(thread_id)
    inputs = {
        "messages": [HumanMessage(content=user_message)],
        "model_id": model_id,
        "problem_description": problem_description,
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
    thread_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Async version of ``stream_answer`` — cancellable via asyncio.

    Use this in async frameworks (FastAPI) so that when the client
    disconnects (e.g. user clicks stop), the asyncio task is cancelled
    and the LLM stream is abandoned immediately.

    Yields:
        str: individual token strings from the synthesizer LLM.
    """
    config = _make_config(thread_id)
    inputs = {
        "messages": [HumanMessage(content=user_message)],
        "model_id": model_id,
        "problem_description": problem_description,
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
    thread_id: str | None = None,
) -> str:
    """Run the full graph and return the final answer as a single string.

    Args:
        graph:               A compiled agent graph (from ``build_graph``).
        model_id:            Equipment model ID provided by the frontend.
        problem_description: Problem description provided by the frontend.
        user_message:        The technician's free-text question.
        thread_id:           Session identifier for persistence.

    Returns:
        The synthesizer's complete answer text.
    """
    config = _make_config(thread_id)
    inputs = {
        "messages": [HumanMessage(content=user_message)],
        "model_id": model_id,
        "problem_description": problem_description,
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
