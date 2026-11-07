#!/usr/bin/env python3
"""Minimal FastAPI shell around the LangGraph RAG turn (Sqlite threads).

  uv sync --group api
  uv run uvicorn api:app --reload --port 8000

  Open http://127.0.0.1:8000/ for a tiny chat UI.
  curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' ^
    -d "{\"question\": \"What is RAG?\", \"thread_id\": \"demo\"}"
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from checkpoint_store import make_sqlite_checkpointer
from graph_rag import (
  build_graph,
  latest_checkpoint_id,
  list_checkpoints,
  run_turn,
)
from index_store import get_vectorstore

# Warm default index once at import (same habit as graph_server).
get_vectorstore(rebuild=False, quiet=True)
_checkpointer = make_sqlite_checkpointer()
graph = build_graph(ensure_index=False, checkpointer=_checkpointer)

app = FastAPI(title='langchain-rag', version='0.1.0')


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


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>langchain-rag chat</title>
  <style>
    :root { color-scheme: light; font-family: 'Segoe UI', system-ui, sans-serif; }
    body { max-width: 44rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }
    h1 { font-size: 1.25rem; margin: 0 0 0.5rem; }
    .meta { color: #555; font-size: 0.9rem; margin-bottom: 1rem; }
    #log { border: 1px solid #ccc; min-height: 12rem; padding: 0.75rem; white-space: pre-wrap; background: #fafafa; }
    #history { border: 1px solid #ddd; max-height: 10rem; overflow: auto; padding: 0.5rem; font-size: 0.85rem; margin: 0.5rem 0; background: #fff; }
    form, .row { display: grid; gap: 0.5rem; margin-top: 0.5rem; }
    input, button { font: inherit; padding: 0.45rem 0.6rem; }
    button { cursor: pointer; }
    .row { grid-template-columns: 1fr auto; align-items: end; }
  </style>
</head>
<body>
  <h1>langchain-rag</h1>
  <p class="meta">
    <code>thread_id</code> = which chat.
    Optional <code>checkpoint_id</code> = resume/fork from that snapshot (leave blank for tip).
  </p>
  <div class="row">
    <label>thread_id <input id="thread" value="default" /></label>
    <button type="button" id="loadHist">Load history</button>
  </div>
  <label>checkpoint_id (optional)
    <input id="ckpt" placeholder="paste id from history…" style="width:100%" />
  </label>
  <div id="history">(click Load history)</div>
  <div id="log"></div>
  <form id="f">
    <input id="q" placeholder="Ask about the notes…" autocomplete="off" required />
    <button type="submit">Ask</button>
  </form>
  <script>
    const log = document.getElementById('log');
    const history = document.getElementById('history');
    const ckpt = document.getElementById('ckpt');

    async function loadHistory() {
      const thread_id = document.getElementById('thread').value.trim() || 'default';
      const res = await fetch('/threads/' + encodeURIComponent(thread_id) + '/checkpoints');
      const data = await res.json();
      if (!res.ok) {
        history.textContent = 'Error: ' + JSON.stringify(data);
        return;
      }
      if (!data.length) {
        history.textContent = '(no checkpoints yet for this thread)';
        return;
      }
      history.innerHTML = data.map((row) => {
        const id = row.checkpoint_id || '';
        return '<div><button type="button" data-id="' + id + '">use</button> '
          + '[' + row.index + '] msgs=' + row.message_count
          + ' <code>' + id + '</code><br/>' + (row.answer_preview || '') + '</div>';
      }).join('<hr/>');
      history.querySelectorAll('button[data-id]').forEach((btn) => {
        btn.addEventListener('click', () => { ckpt.value = btn.getAttribute('data-id') || ''; });
      });
    }

    document.getElementById('loadHist').addEventListener('click', loadHistory);

    document.getElementById('f').addEventListener('submit', async (e) => {
      e.preventDefault();
      const question = document.getElementById('q').value.trim();
      const thread_id = document.getElementById('thread').value.trim() || 'default';
      const checkpoint_id = ckpt.value.trim() || null;
      if (!question) return;
      log.textContent += 'You: ' + question + '\\n';
      document.getElementById('q').value = '';
      const body = { question, thread_id };
      if (checkpoint_id) body.checkpoint_id = checkpoint_id;
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        log.textContent += 'Error: ' + JSON.stringify(data) + '\\n\\n';
        return;
      }
      log.textContent += 'Bot: ' + data.answer
        + '\\n(messages=' + data.message_count
        + ', checkpoint_id=' + (data.checkpoint_id || '') + ')\\n\\n';
      log.scrollTop = log.scrollHeight;
      // After a turn, clear optional fork id so the next ask continues from the new tip.
      ckpt.value = '';
      loadHistory();
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
