#!/usr/bin/env python3
"""LangGraph API / Studio entry: exports compiled RAG graph for langgraph.json.

Do not attach a custom checkpointer here. langgraph.dev provides persistence
itself and rejects graphs compiled with SqliteSaver / MemorySaver.
"""

import settings  # noqa: F401 — load .env before graph import
from graph_rag import build_graph

# Module-level CompiledGraph required by langgraph.json graphs mapping.
graph = build_graph(ensure_index=True)
