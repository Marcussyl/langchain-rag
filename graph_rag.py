#!/usr/bin/env python3
"""LangGraph RAG: retrieve → generate (shared by CLI and Studio)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Checkpointer

from index_store import (
  Citation,
  citation_sources,
  generate_cited_answer,
  get_vectorstore,
  retrieve_context,
)


class RagState(TypedDict):
  """One RAG turn. messages use add_messages so Studio can send only new humans."""

  messages: Annotated[list, add_messages]
  context: str
  citations: list[Citation]
  sources: list[str]
  answer: str


def empty_state() -> RagState:
  return {
    'messages': [],
    'context': '',
    'citations': [],
    'sources': [],
    'answer': '',
  }


def _message_type(msg: Any) -> str | None:
  if isinstance(msg, HumanMessage):
    return 'human'
  msg_type = getattr(msg, 'type', None)
  if isinstance(msg_type, str):
    return msg_type
  if isinstance(msg, dict):
    return msg.get('type') or msg.get('role')
  return None


def _message_content(msg: Any) -> str | None:
  if isinstance(msg, dict):
    content = msg.get('content')
  else:
    content = getattr(msg, 'content', None)
  if content is None:
    return None
  if isinstance(content, str):
    return content
  return str(content)


def last_human_text(messages: list) -> str | None:
  for msg in reversed(messages):
    if _message_type(msg) in {'human', 'user'}:
      return _message_content(msg)
  return None


def retrieve_node(state: RagState) -> dict:
  question = last_human_text(state.get('messages') or [])
  if not question:
    return {'context': '', 'citations': [], 'sources': []}

  context, citations, hits = retrieve_context(question)
  print('(retrieved chunks)')
  for row, doc in zip(citations, hits, strict=True):
    print(f'  [{row["index"]}] {row["source"]}: {row["preview"]}')
  return {
    'context': context,
    'citations': citations,
    'sources': citation_sources(citations),
  }


def generate_node(state: RagState) -> dict:
  question = last_human_text(state.get('messages') or []) or ''
  citations = list(state.get('citations') or [])
  answer = generate_cited_answer(question, state.get('context') or '', citations)
  return {
    'answer': answer,
    'messages': [AIMessage(content=answer)],
  }


def build_graph(*, ensure_index: bool = True, checkpointer: Checkpointer | None = None):
  """Compile retrieve → generate.

  Pass a checkpointer for CLI/API multi-turn threads. Leave None for Studio
  (`graph_server.py`) — langgraph.dev injects its own persistence.
  """
  if ensure_index:
    # Warm the on-disk index so Studio / first CLI turn do not race cold start.
    get_vectorstore(rebuild=False, quiet=True)

  graph = StateGraph(RagState)
  graph.add_node('retrieve', retrieve_node)
  graph.add_node('generate', generate_node)
  graph.add_edge(START, 'retrieve')
  graph.add_edge('retrieve', 'generate')
  graph.add_edge('generate', END)
  return graph.compile(checkpointer=checkpointer)


def thread_config(thread_id: str, checkpoint_id: str | None = None) -> dict:
  """Build LangGraph config. checkpoint_id forks/continues from that snapshot."""
  configurable: dict[str, str] = {'thread_id': thread_id}
  if checkpoint_id:
    configurable['checkpoint_id'] = checkpoint_id
  return {'configurable': configurable}


def list_checkpoints(graph, thread_id: str, *, limit: int = 20) -> list[dict]:
  """Newest-first checkpoint rows for a thread (for CLI / API time travel)."""
  rows: list[dict] = []
  for index, snap in enumerate(graph.get_state_history(thread_config(thread_id))):
    if index >= limit:
      break
    cfg = (snap.config or {}).get('configurable') or {}
    values = snap.values or {}
    answer = str(values.get('answer') or '')
    preview = answer if len(answer) <= 80 else answer[:79] + '…'
    rows.append(
      {
        'index': index,
        'checkpoint_id': cfg.get('checkpoint_id'),
        'message_count': len(values.get('messages') or []),
        'answer_preview': preview,
        'next': list(snap.next or ()),
      }
    )
  return rows


def print_checkpoint_history(graph, thread_id: str, *, limit: int = 12) -> None:
  print(f'(checkpoints for thread_id={thread_id})')
  rows = list_checkpoints(graph, thread_id, limit=limit)
  if not rows:
    print('  (none yet)')
    return
  for row in rows:
    print(
      f'  [{row["index"]}] checkpoint_id={row["checkpoint_id"]}  '
      f'messages={row["message_count"]}  next={row["next"] or ["(done)"]}'
    )
    if row['answer_preview']:
      print(f'       answer: {row["answer_preview"]}')


def latest_checkpoint_id(graph, thread_id: str) -> str | None:
  snap = graph.get_state(thread_config(thread_id))
  cfg = (snap.config or {}).get('configurable') or {}
  return cfg.get('checkpoint_id')


def run_turn(
  graph,
  question: str,
  *,
  thread_id: str | None = None,
  checkpoint_id: str | None = None,
) -> RagState:
  """Invoke one human turn.

  - thread_id only → continue from the thread tip
  - thread_id + checkpoint_id → continue/fork from that snapshot
  """
  payload: dict = {'messages': [HumanMessage(content=question)]}
  if thread_id is None:
    # Stateless one-shot: seed empty fields so nodes always see keys.
    payload = {**empty_state(), **payload}
    return graph.invoke(payload)

  return graph.invoke(payload, thread_config(thread_id, checkpoint_id))
