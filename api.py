#!/usr/bin/env python3
"""Minimal FastAPI shell around the LangGraph RAG turn.

  uv sync --group api
  uv run uvicorn api:app --reload --port 8000

  curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' ^
    -d "{\"question\": \"What is RAG?\"}"
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from graph_rag import build_graph, run_turn
from index_store import get_vectorstore

# Warm default index once at import (same habit as graph_server).
get_vectorstore(rebuild=False, quiet=True)
graph = build_graph(ensure_index=False)

app = FastAPI(title='langchain-rag', version='0.1.0')


class AskRequest(BaseModel):
  question: str = Field(min_length=1, description='User question')


class AskResponse(BaseModel):
  answer: str
  sources: list[str]
  citations: list[dict]


@app.get('/health')
def health() -> dict[str, str]:
  return {'status': 'ok'}


@app.post('/ask', response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
  result = run_turn(graph, body.question.strip())
  return AskResponse(
    answer=result.get('answer') or '',
    sources=list(result.get('sources') or []),
    citations=list(result.get('citations') or []),
  )
