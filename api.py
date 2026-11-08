#!/usr/bin/env python3
"""Minimal FastAPI shell around the LangGraph RAG turn (Sqlite threads).

  uv sync --group api
  uv run uvicorn api:app --reload --port 8000

  Open http://127.0.0.1:8000/ for chat, /library for the knowledge base.
  curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' ^
    -d "{\"question\": \"What is RAG?\", \"thread_id\": \"demo\"}"
"""

from __future__ import annotations

import html
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

import settings  # noqa: F401 — load .env before checkpointer / graph
from checkpoint_store import close_checkpointer_resources, database_url, make_checkpointer
from graph_rag import (
  build_graph,
  latest_checkpoint_id,
  list_checkpoints,
  run_turn,
  thread_config,
)
from index_store import (
  ALLOWED_UPLOAD_SUFFIXES,
  DEFAULT_CONFIG,
  MAX_UPLOAD_BYTES,
  OLLAMA_EMBED_MODEL,
  delete_note_file,
  get_vectorstore,
  ingest_file,
  list_note_files,
  save_upload_bytes,
)

# Warm default index once at import (same habit as graph_server).
get_vectorstore(rebuild=False, quiet=True)
_checkpointer = make_checkpointer()
graph = build_graph(ensure_index=False, checkpointer=_checkpointer)


@asynccontextmanager
async def lifespan(_app: FastAPI):
  yield
  close_checkpointer_resources()


app = FastAPI(title='langchain-rag', version='0.1.0', lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / 'static'
CHAT_HTML_PATH = STATIC_DIR / 'chat.html'
LIBRARY_HTML_PATH = STATIC_DIR / 'library.html'


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
  question_preview: str = ''
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


class DocumentRow(BaseModel):
  source: str
  filename: str
  kind: str
  suffix: str
  bytes: int
  mtime: int
  chunks: int


class DocumentListResponse(BaseModel):
  documents: list[DocumentRow]
  count: int


class DeleteDocumentResponse(BaseModel):
  source: str
  deleted_file: bool
  removed_chunks: int


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


def _fill_common(template: str) -> str:
  checkpointer = 'postgres checkpointer' if database_url() else 'sqlite checkpointer'
  doc_count = str(len(list_note_files()))
  return (
    template
    .replace('{{CHAT_BACKEND}}', html.escape(_chat_backend_label()))
    .replace('{{CHECKPOINTER}}', html.escape(checkpointer))
    .replace('{{EMBED_MODEL}}', html.escape(OLLAMA_EMBED_MODEL))
    .replace('{{RETRIEVE_K}}', html.escape(str(DEFAULT_CONFIG.k)))
    .replace('{{SEARCH_TYPE}}', html.escape(DEFAULT_CONFIG.search_type))
    .replace('{{DOC_COUNT}}', html.escape(doc_count))
  )


def render_chat_html() -> str:
  """Fill static/chat.html with live backend labels."""
  return _fill_common(CHAT_HTML_PATH.read_text(encoding='utf-8'))


def render_library_html() -> str:
  """Fill static/library.html with live backend labels."""
  return _fill_common(LIBRARY_HTML_PATH.read_text(encoding='utf-8'))


def _sse(payload: dict) -> str:
  return f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'


@app.get('/', response_class=HTMLResponse)
def chat_page() -> str:
  if not CHAT_HTML_PATH.is_file():
    raise HTTPException(status_code=500, detail=f'missing UI template: {CHAT_HTML_PATH}')
  return render_chat_html()


@app.get('/library', response_class=HTMLResponse)
def library_page() -> str:
  if not LIBRARY_HTML_PATH.is_file():
    raise HTTPException(status_code=500, detail=f'missing UI template: {LIBRARY_HTML_PATH}')
  return render_library_html()


@app.get('/health')
def health() -> dict[str, str]:
  return {'status': 'ok'}


@app.get('/documents', response_model=DocumentListResponse)
def get_documents() -> DocumentListResponse:
  rows = [DocumentRow(**row) for row in list_note_files()]
  return DocumentListResponse(documents=rows, count=len(rows))


@app.delete('/documents/{source:path}', response_model=DeleteDocumentResponse)
def delete_document(source: str) -> DeleteDocumentResponse:
  """Delete a note under data/ and remove matching Chroma chunks."""
  label = unquote(source).strip()
  try:
    info = delete_note_file(label)
  except FileNotFoundError as exc:
    raise HTTPException(status_code=404, detail=str(exc)) from exc
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  return DeleteDocumentResponse(**info)


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


@app.post('/ask/stream')
def ask_stream(body: AskRequest) -> StreamingResponse:
  """SSE: token chunks + final answer/citations (keeps /ask for non-streaming clients)."""
  thread_id = body.thread_id.strip() or 'default'
  checkpoint_id = (body.checkpoint_id or '').strip() or None
  question = body.question.strip()
  payload = {'messages': [HumanMessage(content=question)]}
  config = thread_config(thread_id, checkpoint_id)

  def event_gen():
    yield _sse({'type': 'status', 'status': 'running'})
    final_values: dict | None = None
    try:
      for mode, chunk in graph.stream(
        payload,
        config,
        stream_mode=['custom', 'updates', 'values'],
      ):
        if mode == 'custom' and isinstance(chunk, dict) and chunk.get('type') == 'token':
          yield _sse({'type': 'token', 'text': str(chunk.get('text') or '')})
        elif mode == 'updates' and isinstance(chunk, dict) and 'retrieve' in chunk:
          retrieved = chunk['retrieve'] or {}
          yield _sse(
            {
              'type': 'retrieved',
              'citations': list(retrieved.get('citations') or []),
              'sources': list(retrieved.get('sources') or []),
            }
          )
        elif mode == 'values' and isinstance(chunk, dict):
          final_values = chunk
      if not final_values:
        yield _sse({'type': 'error', 'detail': 'stream ended without values'})
        return
      messages = final_values.get('messages') or []
      yield _sse(
        {
          'type': 'done',
          'answer': final_values.get('answer') or '',
          'sources': list(final_values.get('sources') or []),
          'citations': list(final_values.get('citations') or []),
          'thread_id': thread_id,
          'message_count': len(messages),
          'checkpoint_id': latest_checkpoint_id(graph, thread_id),
        }
      )
    except Exception as exc:  # noqa: BLE001 — surface to SSE client
      yield _sse({'type': 'error', 'detail': str(exc)})

  return StreamingResponse(
    event_gen(),
    media_type='text/event-stream',
    headers={
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
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
