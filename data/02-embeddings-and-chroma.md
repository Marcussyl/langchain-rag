# Embeddings and Chroma in this project

## Embeddings

An embedding model turns text into a list of numbers (a vector) so similar
meanings land near each other in vector space.

This repo defaults to Ollama:

- Chat: `llama3.2` (override with `OLLAMA_MODEL`)
- Embed: `nomic-embed-text` (override with `OLLAMA_EMBED_MODEL`)

You must pull both models before running `rag.py`.

## Chroma

Chroma is a local vector store. Here it persists under `.chroma/` so indexes
survive process restarts (the folder is gitignored).

Collection name used by `rag.py`: `tutor-notes`.

## When to consider Postgres later

Use PostgreSQL + pgvector when you need shared production storage, backups,
or the same database for app data and vectors. For learning RAG locally,
Chroma is enough.
