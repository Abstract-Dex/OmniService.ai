# OmniService.ai

Agentic HVAC/R field-service assistant. Ingest service manuals into a hybrid knowledge base (vector + graph), answer technician questions via streaming chat, persist conversations per project, and generate end-of-session reports.

---

## What It Does

- **Ingest PDF manuals** → Weaviate (vector) + Neo4j (graph)
- **Chat API** → LangGraph agent streams answers from KB + optional web search
- **Project persistence** → Per-`project_id` conversation state, technician handoff
- **View sources** → Citations (manual + page) + serve PDFs for inline viewer
- **End chat report** → LLM-generated markdown summary of problem, troubleshooting, resolution

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FastAPI (api.py)                                                           │
│  POST /api/chat         → stream answer                                     │
│  POST /api/chat/report  → end-chat markdown report                          │
│  GET  /api/projects     → list projects for user                            │
│  GET  /api/projects/{id}/history → messages + references                    │
│  GET  /api/manuals/{name} → serve PDF                                       │
│  DELETE /api/projects/{id} → delete project (user_id required)              │
└─────────────────────────────────────────────────────────────────────────────┘
        │                    │                          │
        ▼                    ▼                          ▼
┌──────────────┐   ┌───────────────────┐   ┌──────────────────────┐
│  persistence │   │  LangGraph Agent  │   │  search_knowledge_   │
│  (SQLite)    │   │  (modules/agent)  │   │  base (Weaviate +    │
│  projects,   │   │  planner→executor │   │  Neo4j + Cohere)     │
│  users,      │   │  →web_search?     │   │                      │
│  preferences │   │  →synthesizer     │   └──────────────────────┘
└──────────────┘   └───────────────────┘
        │                    │                          │
        │                    │                          │
        ▼                    ▼                          ▼
┌──────────────┐   ┌───────────────────┐   ┌─────────────────────┐
│  app_state.  │   │  checkpoints.db   │   │  Weaviate (vector)  │
│  db          │   │  (LangGraph       │   │  Neo4j (graph)      │
│              │   │  thread state)    │   │  Cohere             │
└──────────────┘   └───────────────────┘   └─────────────────────┘
```

### Agent Flow

```
START → planner → executor ─┬─ (web_search=true)  → web_search → synthesizer → END
                           └─ (web_search=false) → synthesizer → END
```

---

## File Roles

| File                         | Role                                                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| **api.py**                   | FastAPI app, CORS, chat/report/projects/manuals/delete endpoints                                |
| **modules/agent.py**         | LangGraph graph (planner, executor, web_search, synthesizer), report generator                  |
| **modules/template.py**      | All LLM prompts (planner, synthesizer, web_search, report)                                      |
| **modules/search.py**        | `search_knowledge_base()` — Weaviate hybrid + Cohere rerank + Neo4j subgraph; `web_search_tool` |
| **modules/persistence.py**   | SQLite: projects, project_users, user_preferences; handoff, delete                              |
| **modules/vector_ingest.py** | Cohere embed → Weaviate `service_manuals` (page chunks, manual_name)                            |
| **modules/graph_store.py**   | Neo4j: Model, Component, ConflictRule, Troubleshooting; `ingest_sbom()`                         |
| **modules/ingest.py**        | Pipeline: PDF → extract → vector_ingest → llm_parser → graph_store                              |
| **modules/extract.py**       | pdfplumber: PDF → JSON (raw_text, tables per page)                                              |
| **modules/llm_parser.py**    | Claude Haiku: raw JSON → UnifiedModel (sBOM, specs, rules)                                      |
| **modules/scheme.py**        | Pydantic schemas for ingestion (UnifiedModel, etc.)                                             |

---
