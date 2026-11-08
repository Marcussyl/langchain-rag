# What is RAG?

Retrieval Augmented Generation (RAG) is a pattern where an application
**retrieves** relevant text from an external knowledge base, then **generates**
an answer with an LLM using that text as context.

## Why use RAG?

- Model weights alone may be outdated or missing private docs.
- You can cite sources (which file / chunk was retrieved).
- Updating knowledge often means re-indexing documents, not fine-tuning.

## Typical pipeline

1. Load documents
2. Split into chunks
3. Embed chunks into vectors
4. Store vectors in a vector database
5. At query time: embed the question → similarity search → LLM answer

## Not the same as chat memory

Chat compression (summaries / trim) keeps a **conversation** short.
RAG retrieves from a **document corpus**. Both can appear in one agent, but
they solve different problems.
