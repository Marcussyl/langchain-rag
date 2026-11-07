# Local RAG sample notes

## What is RAG?

RAG means **Retrieval Augmented Generation**. The app first retrieves relevant
text chunks from a knowledge base, then asks the LLM to answer using that
context instead of relying only on the model weights.

## Stack in this repo

- **Embeddings**: Ollama `nomic-embed-text` (local)
- **Vector store**: Chroma on disk under `.chroma/`
- **Chat model**: Ollama `llama3.2` (or `OLLAMA_MODEL` from the environment)

## Why not Postgres yet?

PostgreSQL / pgvector is useful later for production persistence. For learning
RAG locally, Chroma keeps the setup small.
