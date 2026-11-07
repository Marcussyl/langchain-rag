#!/usr/bin/env python3
"""LangGraph RAG: retrieve → generate (shared by CLI and Studio)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

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


def build_graph(*, ensure_index: bool = True):
  """Compile retrieve → generate. ensure_index builds/opens .chroma once at compile time."""
  if ensure_index:
    # Warm the on-disk index so Studio / first CLI turn do not race cold start.
    get_vectorstore(rebuild=False, quiet=True)

  graph = StateGraph(RagState)
  graph.add_node('retrieve', retrieve_node)
  graph.add_node('generate', generate_node)
  graph.add_edge(START, 'retrieve')
  graph.add_edge('retrieve', 'generate')
  graph.add_edge('generate', END)
  return graph.compile()


def run_turn(graph, question: str) -> RagState:
  result = graph.invoke(
    {
      **empty_state(),
      'messages': [HumanMessage(content=question)],
    }
  )
  return result
