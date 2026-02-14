# OmniService.ai

Agentic system for HVAC/R field-service AI: ingests service manuals into a hybrid knowledge base (vector + graph) and answers technician questions via a LangGraph-powered chat API.

---

## What Has Been Done

### Phase 1: Knowledge Base Ingestion

| Component                                         | Description                                                                                                             |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **PDF extraction** (`modules/extract.py`)         | pdfplumber extracts raw text and tables per page → JSON                                                                 |
| **Pydantic schemas** (`modules/scheme.py`)        | `UnifiedModel`, `Classification`, `Specification`, `SBOM`, `ConflictRules`, `Troubleshooting`                           |
| **LLM parser** (`modules/llm_parser.py`)          | Claude Haiku parses extracted JSON → validated `UnifiedModel`, code-fence stripping                                     |
| **Vector ingestion** (`modules/vector_ingest.py`) | Cohere embed-v4.0, Weaviate `service_manuals` collection (hybrid BM25 + vector), per-page chunks                        |
| **Graph ingestion** (`modules/graph_store.py`)    | Neo4j nodes (Model, Component, ConflictRule, Troubleshooting), recursive sBOM, `SHARED_PART` cross-model links          |
| **Unified pipeline** (`modules/ingest.py`)        | PDF → extract → vector ingest → LLM parse → graph ingest; skips LLM when model already in Neo4j                         |
| **Search** (`modules/search.py`)                  | `search_knowledge_base(model_id, problem_description)` — hybrid search + Cohere rerank + Neo4j subgraph → combined JSON |
| **Docker Compose**                                | `docker-compose-weaviate.yml`, `docker-compose-graph.yaml` for local Weaviate and Neo4j                                 |

### Phase 2: Agentic Answer Generation

| Component                            | Description                                                                                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Prompts** (`modules/template.py`)  | Centralised prompts for planner and synthesizer (markdown formatting for frontend)                                                                     |
| **Agent graph** (`modules/agent.py`) | LangGraph `[START] → planner → executor → synthesizer → [END]` with persistence, durable execution, streaming                                          |
| **Planner node**                     | Decides which tools to call (`search_knowledge_base`, `external_search`) based on user message; frontend provides `model_id` and `problem_description` |
| **Executor node**                    | Calls `search_knowledge_base` (and future `external_search`); side effects wrapped in `@task` for durable execution                                    |
| **Synthesizer node**                 | Generates technician-facing answers from KB results; outputs markdown (bold, headings, lists, blockquotes)                                             |
| **Conversation memory**              | Per `project_id` (thread) via LangGraph checkpointer; only user + assistant messages stored                                                            |
| **API** (`api.py`)                   | FastAPI `POST /api/chat` — SSE streaming, CORS, client-disconnect detection (stop button)                                                              |

---

## Project Structure

```
OmniService.ai/
├── api.py                    # FastAPI server — POST /api/chat
├── modules/
│   ├── __init__.py
│   ├── scheme.py             # Pydantic schemas
│   ├── extract.py             # PDF → JSON (pdfplumber)
│   ├── llm_parser.py          # JSON → UnifiedModel (Claude Haiku)
│   ├── vector_ingest.py       # Cohere + Weaviate ingestion
│   ├── graph_store.py         # Neo4j ingestion + queries
│   ├── ingest.py              # Full ingestion pipeline
│   ├── search.py              # search_knowledge_base()
│   ├── template.py            # Agent prompts (planner, synthesizer)
│   └── agent.py               # LangGraph graph (planner, executor, synthesizer)
├── docker-compose-weaviate.yml
├── docker-compose-graph.yaml
├── test.ipynb                # Demo notebook
└── pyproject.toml
```

---

## Running the API

### 1. Start infrastructure

```bash
docker compose -f docker-compose-weaviate.yml up -d
docker compose -f docker-compose-graph.yaml up -d
```

### 2. Set environment variables

Create a `.env` with:

- `ANTHROPIC_API_KEY` — for the agent LLM
- `COHERE_API_KEY` — for embeddings and rerank
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (optional; defaults: `neo4j://localhost:7687`, `neo4j`, `omniservice`)

### 3. Run the API server

```bash
uvicorn api:app --reload --port 8000
```

- **Swagger UI:** http://localhost:8000/docs
- **Endpoint:** `POST http://localhost:8000/api/chat`

### 4. Request format

```json
{
  "project_id": "abc123",
  "device_type": "Refrigeration Unit",
  "model_number": "CPB050JC-S-0-EV",
  "problem_description": "Unit not cooling properly, compressor cycling on and off",
  "messages": [
    {
      "role": "user",
      "content": "I'm having an issue with my Refrigeration Unit..."
    },
    { "role": "assistant", "content": "Based on your device..." },
    { "role": "user", "content": "I'm seeing error code E3" }
  ]
}
```

- The last `user` message is treated as the current question.
- `project_id` is used as LangGraph `thread_id` (conversation persistence).
- `model_number` maps to the knowledge-base `model_id`.

### 5. Response

Streamed via **Server-Sent Events** (`text/event-stream`): tokens streamed one-by-one, ending with `[DONE]`. The frontend can abort the request (e.g. stop button) and the server stops generating.

---

## Ingestion Pipeline

To ingest PDF manuals into the knowledge base:

```bash
python -m modules.ingest -i Data/
```

Expects `Data/*.pdf`. Outputs go to `Data/extracted/` and `Data/parsed/`, then into Weaviate and Neo4j.

---

## Dependencies

See `pyproject.toml`. Key: FastAPI, uvicorn, langgraph, langchain-anthropic, langchain-core, cohere, weaviate-client, neo4j, anthropic, pdfplumber, pydantic.
