# langchain-rag

Separate learning repo for **Phase 7** themes that do not belong in
[`langchain-first-chain`](../langchain-first-chain):

- RAG / vector stores / document chunking
- LangGraph `retrieve` → `generate` + local Studio (`langgraph.dev`)
- Retrieval presets (`chunk_size` / `k` / MMR) + FastAPI `/ask` + tiny chat UI
- Thread checkpointer for multi-turn CLI/API: Sqlite by default, optional Postgres
- Optional later: Agent Server / `langgraph deploy`

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Setup

```bash
uv sync --group dev
```

Indexes live under `.chroma/c{chunk}_o{overlap}/` (gitignored). After pulling newer
chunk settings, run once with `--rebuild` (or `compare_retrieve.py --rebuild`).

## 1. Starter RAG (LCEL, no graph)

```bash
uv run python rag.py                 # interactive Q&A (reuses index)
uv run python rag.py --rebuild       # re-index data/ then chat
uv run python rag.py -q 'What is RAG?'   # one-shot only when you pass -q
```

Pipeline: load `data/*.md` → split → embed with Ollama → Chroma → answer.

## 2. LangGraph RAG (retrieve → generate)

```bash
uv run python graph_chat.py
uv run python graph_chat.py --rebuild
uv run python graph_chat.py -q 'What is RAG?'
```

Answers use inline `[n]` cites when the model quotes a chunk, plus a fixed
`Sources:` footer (filename + preview) appended in Python — not invented by the LLM.

```mermaid
flowchart LR
  START([START]) --> retrieve
  retrieve --> generate
  generate --> END([END])
```

## 3. Studio (`langgraph.dev`)

```bash
uv sync --group dev
uv run langgraph dev
```

- API default: `http://localhost:2024`
- Graph name in Studio: **`rag`**
- Do **not** attach SqliteSaver in `graph_server.py` (API owns persistence)

## 4. Compare retrieval presets

Same question, four presets (baseline / larger chunks / higher `k` / MMR). Default
prints **Sources only** (no chat LLM):

```bash
uv run python compare_retrieve.py -q 'What is RAG?'
uv run python compare_retrieve.py -q 'What is RAG?' --rebuild
uv run python compare_retrieve.py -q 'What is RAG?' --answer
```

Scan which filenames/previews look most relevant before changing defaults in `.env`.

## 5. FastAPI shell + chat page

```bash
uv sync --group api
uv run uvicorn api:app --reload --port 8000
```

- Chat UI: http://127.0.0.1:8000/ (Load history → **use** a `checkpoint_id`)
- `GET /health`
- `GET /threads/{thread_id}/checkpoints`
- `POST /ask` with optional `checkpoint_id` to fork from that snapshot

Same `thread_id` reuses Sqlite checkpoints (`checkpoints.sqlite`).  
`graph_server.py` stays checkpointer-free so Studio can inject its own.

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is RAG?\", \"thread_id\": \"demo\"}"
curl http://127.0.0.1:8000/threads/demo/checkpoints
```

CLI: `history` lists ids; `use <checkpoint_id>` pins the next turn; or `--checkpoint <id>`.

## 6. Optional Postgres checkpointer

Default persistence is `checkpoints.sqlite`. To use Postgres instead:

```bash
docker compose up -d
# PowerShell:
$env:DATABASE_URL = 'postgresql://rag:rag@127.0.0.1:5432/rag_checkpoints'
uv run python graph_chat.py --thread pg-demo -q 'What is RAG?'
uv run uvicorn api:app --reload --port 8000
```

Unset `DATABASE_URL` to fall back to Sqlite. Studio still uses its own store.

## Env

Optional (see `.env.example`): `OLLAMA_*`, `RAG_CHUNK_*`, `RAG_RETRIEVE_K`,
`RAG_SEARCH_TYPE` (`similarity` | `mmr`), `DATABASE_URL` (Postgres checkpointer).

A leftover PowerShell `$env:RAG_QUERY` is **ignored** unless you pass `-q`.

## Layout

| Path | Role |
|------|------|
| `index_store.py` | Shared load / Chroma / citations / `RagConfig` |
| `rag.py` | LCEL RAG demo (no LangGraph) |
| `graph_rag.py` | StateGraph: retrieve → generate |
| `graph_server.py` | Studio export (`langgraph.json`) |
| `graph_chat.py` | CLI for the graph |
| `compare_retrieve.py` | Side-by-side Sources for presets |
| `api.py` | FastAPI `/` chat UI + `POST /ask` |
| `checkpoint_store.py` | SqliteSaver or PostgresSaver for CLI/API |
| `docker-compose.yml` | Local Postgres for checkpoints |
| `langgraph.json` | `langgraph.dev` config |
| `data/` | Source documents to index |
| `.chroma/` | Persistent Chroma indexes (gitignored) |
| `checkpoints.sqlite` | Default thread history (gitignored) |
