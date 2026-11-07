#!/usr/bin/env python3
"""Minimal FastAPI shell around the LangGraph RAG turn (Sqlite threads).

  uv sync --group api
  uv run uvicorn api:app --reload --port 8000

  Open http://127.0.0.1:8000/ for a tiny chat UI.
  curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' ^
    -d "{\"question\": \"What is RAG?\", \"thread_id\": \"demo\"}"
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from checkpoint_store import make_sqlite_checkpointer
from graph_rag import build_graph, run_turn
from index_store import get_vectorstore

# Warm default index once at import (same habit as graph_server).
get_vectorstore(rebuild=False, quiet=True)
_checkpointer = make_sqlite_checkpointer()
graph = build_graph(ensure_index=False, checkpointer=_checkpointer)

app = FastAPI(title='langchain-rag', version='0.1.0')


class AskRequest(BaseModel):
  question: str = Field(min_length=1, description='User question')
  thread_id: str = Field(default='default', min_length=1, description='Conversation thread')


class AskResponse(BaseModel):
  answer: str
  sources: list[str]
  citations: list[dict]
  thread_id: str
  message_count: int


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>langchain-rag chat</title>
  <style>
    :root { color-scheme: light; font-family: 'Segoe UI', system-ui, sans-serif; }
    body { max-width: 42rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }
    h1 { font-size: 1.25rem; margin: 0 0 0.5rem; }
    .meta { color: #555; font-size: 0.9rem; margin-bottom: 1rem; }
    #log { border: 1px solid #ccc; min-height: 14rem; padding: 0.75rem; white-space: pre-wrap; background: #fafafa; }
    form { display: grid; gap: 0.5rem; margin-top: 0.75rem; }
    input, button { font: inherit; padding: 0.45rem 0.6rem; }
    button { cursor: pointer; }
  </style>
</head>
<body>
  <h1>langchain-rag</h1>
  <p class="meta">Same <code>thread_id</code> keeps checkpointed history (Sqlite). Change it for a fresh chat.</p>
  <label>thread_id <input id="thread" value="default" /></label>
  <div id="log"></div>
  <form id="f">
    <input id="q" placeholder="Ask about the notes…" autocomplete="off" required />
    <button type="submit">Ask</button>
  </form>
  <script>
    const log = document.getElementById('log');
    const form = document.getElementById('f');
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const question = document.getElementById('q').value.trim();
      const thread_id = document.getElementById('thread').value.trim() || 'default';
      if (!question) return;
      log.textContent += 'You: ' + question + '\\n';
      document.getElementById('q').value = '';
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, thread_id }),
      });
      const data = await res.json();
      if (!res.ok) {
        log.textContent += 'Error: ' + JSON.stringify(data) + '\\n\\n';
        return;
      }
      log.textContent += 'Bot: ' + data.answer + '\\n(messages in thread: ' + data.message_count + ')\\n\\n';
      log.scrollTop = log.scrollHeight;
    });
  </script>
</body>
</html>
"""


@app.get('/', response_class=HTMLResponse)
def chat_page() -> str:
  return CHAT_HTML


@app.get('/health')
def health() -> dict[str, str]:
  return {'status': 'ok'}


@app.post('/ask', response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
  thread_id = body.thread_id.strip() or 'default'
  result = run_turn(graph, body.question.strip(), thread_id=thread_id)
  messages = result.get('messages') or []
  return AskResponse(
    answer=result.get('answer') or '',
    sources=list(result.get('sources') or []),
    citations=list(result.get('citations') or []),
    thread_id=thread_id,
    message_count=len(messages),
  )
