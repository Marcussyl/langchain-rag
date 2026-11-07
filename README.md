# langchain-rag

Separate learning repo for **Phase 7** themes that do not belong in
[`langchain-first-chain`](../langchain-first-chain):

- RAG / vector stores / document chunking
- Optional later: Agent Server deploy, product UI, Postgres checkpointers

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Setup

```bash
uv sync
```

## Run the starter RAG script

```bash
uv run python rag.py
```

Pipeline: load `data/*.md` → split → embed with Ollama → Chroma (`.chroma/`) → answer.

Optional env (see `.env.example`): `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `RAG_QUERY`.

## Layout

| Path | Role |
|------|------|
| `rag.py` | End-to-end local RAG demo |
| `data/` | Source documents to index |
| `src/langchain_rag/` | Package stub from `uv init` |
| `.chroma/` | Persistent Chroma index (gitignored) |
