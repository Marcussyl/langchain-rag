#!/usr/bin/env python3
"""Minimal FastAPI shell around the LangGraph RAG turn (Sqlite threads).

  uv sync --group api
  uv run uvicorn api:app --reload --port 8000

  Open http://127.0.0.1:8000/ for the notes chat UI (Stitch-inspired workbench).
  curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' ^
    -d "{\"question\": \"What is RAG?\", \"thread_id\": \"demo\"}"
"""

from __future__ import annotations

import html
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import settings  # noqa: F401 — load .env before checkpointer / graph
from checkpoint_store import database_url, make_checkpointer
from graph_rag import (
  build_graph,
  latest_checkpoint_id,
  list_checkpoints,
  run_turn,
)
from index_store import (
  ALLOWED_UPLOAD_SUFFIXES,
  DEFAULT_CONFIG,
  MAX_UPLOAD_BYTES,
  OLLAMA_EMBED_MODEL,
  get_vectorstore,
  ingest_file,
  save_upload_bytes,
)

# Warm default index once at import (same habit as graph_server).
get_vectorstore(rebuild=False, quiet=True)
_checkpointer = make_checkpointer()
graph = build_graph(ensure_index=False, checkpointer=_checkpointer)

app = FastAPI(title='langchain-rag', version='0.1.0')

CHAT_HTML_PATH = Path(__file__).resolve().parent / 'static' / 'chat.html'


class AskRequest(BaseModel):
  question: str = Field(min_length=1, description='User question')
  thread_id: str = Field(default='default', min_length=1, description='Conversation thread')
  checkpoint_id: str | None = Field(
    default=None,
    description='Optional: fork/continue from this snapshot instead of the thread tip',
  )


class AskResponse(BaseModel):
  answer: str
  sources: list[str]
  citations: list[dict]
  thread_id: str
  message_count: int
  checkpoint_id: str | None = None


class CheckpointRow(BaseModel):
  index: int
  checkpoint_id: str | None
  message_count: int
  answer_preview: str
  next: list[str]


class UploadItemResult(BaseModel):
  filename: str
  source: str
  chunks: int
  ok: bool
  error: str | None = None


class UploadResponse(BaseModel):
  files: list[UploadItemResult]
  indexed: int


def _chat_backend_label() -> str:
  backend = (os.getenv('CHAT_BACKEND') or 'ollama').strip().lower() or 'ollama'
  if backend == 'openrouter':
    model = (os.getenv('OPENROUTER_MODEL') or 'openrouter').strip() or 'openrouter'
    short = model.split('/')[-1]
    if len(short) > 28:
      short = short[:27] + '…'
    return f'openrouter · {short}'
  if backend == 'gemini':
    return (os.getenv('GEMINI_MODEL') or 'gemini').strip() or 'gemini'
  if backend == 'groq':
    return (os.getenv('GROQ_MODEL') or 'groq').strip() or 'groq'
  return (os.getenv('OLLAMA_MODEL') or 'ollama').strip() or 'ollama'


def render_chat_html() -> str:
  """Fill static/chat.html with live backend labels (skip unimplemented Stitch chrome)."""
  template = CHAT_HTML_PATH.read_text(encoding='utf-8')
  checkpointer = 'postgres checkpointer' if database_url() else 'sqlite checkpointer'
  return (
    template
    .replace('{{CHAT_BACKEND}}', html.escape(_chat_backend_label()))
    .replace('{{CHECKPOINTER}}', html.escape(checkpointer))
    .replace('{{EMBED_MODEL}}', html.escape(OLLAMA_EMBED_MODEL))
    .replace('{{RETRIEVE_K}}', html.escape(str(DEFAULT_CONFIG.k)))
    .replace('{{SEARCH_TYPE}}', html.escape(DEFAULT_CONFIG.search_type))
  )


@app.get('/', response_class=HTMLResponse)
def chat_page() -> str:
  if not CHAT_HTML_PATH.is_file():
    raise HTTPException(status_code=500, detail=f'missing UI template: {CHAT_HTML_PATH}')
  return render_chat_html()


@app.get('/health')
def health() -> dict[str, str]:
  return {'status': 'ok'}


@app.get('/threads/{thread_id}/checkpoints', response_model=list[CheckpointRow])
def get_checkpoints(thread_id: str, limit: int = 20) -> list[CheckpointRow]:
  tid = thread_id.strip() or 'default'
  if limit < 1 or limit > 100:
    raise HTTPException(status_code=400, detail='limit must be 1..100')
  rows = list_checkpoints(graph, tid, limit=limit)
  return [CheckpointRow(**row) for row in rows]


@app.post('/ask', response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
  thread_id = body.thread_id.strip() or 'default'
  checkpoint_id = (body.checkpoint_id or '').strip() or None
  result = run_turn(
    graph,
    body.question.strip(),
    thread_id=thread_id,
    checkpoint_id=checkpoint_id,
  )
  messages = result.get('messages') or []
  return AskResponse(
    answer=result.get('answer') or '',
    sources=list(result.get('sources') or []),
    citations=list(result.get('citations') or []),
    thread_id=thread_id,
    message_count=len(messages),
    checkpoint_id=latest_checkpoint_id(graph, thread_id),
  )


@app.post('/upload', response_model=UploadResponse)
async def upload_notes(files: list[UploadFile] = File(...)) -> UploadResponse:
  """Save notes under data/uploads/ and incrementally index into Chroma."""
  if not files:
    raise HTTPException(status_code=400, detail='no files uploaded')

  results: list[UploadItemResult] = []
  indexed = 0
  for upload in files:
    name = upload.filename or 'unnamed'
    try:
      data = await upload.read()
      if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f'file too large (max {MAX_UPLOAD_BYTES} bytes)')
      path = save_upload_bytes(name, data)
      info = ingest_file(path, quiet=True)
      indexed += 1
      results.append(
        UploadItemResult(
          filename=name,
          source=str(info['source']),
          chunks=int(info['chunks']),
          ok=True,
        )
      )
    except Exception as exc:  # noqa: BLE001 — per-file errors returned to UI
      results.append(
        UploadItemResult(
          filename=name,
          source='',
          chunks=0,
          ok=False,
          error=str(exc),
        )
      )

  if indexed == 0:
    raise HTTPException(
      status_code=400,
      detail={
        'message': 'no files indexed',
        'allowed': sorted(ALLOWED_UPLOAD_SUFFIXES),
        'files': [item.model_dump() for item in results],
      },
    )
  return UploadResponse(files=results, indexed=indexed)
