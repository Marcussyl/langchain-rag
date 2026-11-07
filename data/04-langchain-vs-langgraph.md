# LangChain vs LangGraph (quick map)

## LangChain

Good for LCEL chains, tools, RAG retrieve-then-generate, and high-level
helpers such as `create_agent`.

## LangGraph

Good when you need an explicit state machine: conditional edges, tool loops,
HITL interrupts, checkpointed threads, and streaming modes.

## How this workspace splits learning

- `langchain-first-chain` — tutor graph (memory, tools, HITL, stream, traces).
- `langchain-rag` — document retrieval and vector stores.

You can later wire a LangGraph node that calls a retriever, but keep the first
RAG experiments as a straight script so the pipeline stays obvious.
