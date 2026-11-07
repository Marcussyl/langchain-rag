#!/usr/bin/env python3
"""CLI for the LangGraph RAG pipeline (retrieve → generate + Sqlite threads).

Run from repo root:
  uv run python graph_chat.py
  uv run python graph_chat.py --rebuild
  uv run python graph_chat.py -q 'What is RAG?'
  uv run python graph_chat.py --thread demo

Studio (same graph, platform checkpointer — no Sqlite here):
  uv run langgraph dev
"""

from __future__ import annotations

import argparse
import os

from checkpoint_store import make_sqlite_checkpointer
from graph_rag import build_graph, run_turn
from index_store import get_vectorstore


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='LangGraph RAG chat (Ollama + Chroma)')
  parser.add_argument(
    '-q',
    '--query',
    default=None,
    help='One-shot question (skip interactive loop).',
  )
  parser.add_argument(
    '--rebuild',
    action='store_true',
    help='Delete .chroma index for the default chunk config and re-index data/',
  )
  parser.add_argument(
    '--thread',
    default='default',
    help='Checkpoint thread_id (same id continues the conversation).',
  )
  return parser.parse_args()


def print_result(result: dict) -> None:
  # Full answer already includes the deterministic Sources footer.
  print('Bot:', result.get('answer') or '(no answer)')
  messages = result.get('messages') or []
  print(f'(thread messages: {len(messages)})')


def main() -> None:
  args = parse_args()
  get_vectorstore(rebuild=args.rebuild)
  checkpointer = make_sqlite_checkpointer()
  graph = build_graph(ensure_index=False, checkpointer=checkpointer)
  thread_id = args.thread.strip() or 'default'

  if args.query:
    print(f'(query) {args.query}')
    print(f'(thread) {thread_id}')
    print()
    print_result(run_turn(graph, args.query, thread_id=thread_id))
    return

  leftover = os.getenv('RAG_QUERY')
  if leftover:
    print('(note: RAG_QUERY is set in the shell but ignored; use -q to one-shot, or clear it)')

  print('Interactive LangGraph RAG (retrieve → generate). Empty input ignored; type q to quit.')
  print(f'thread_id={thread_id} (Sqlite checkpointer). Studio: uv run langgraph dev')
  print()
  while True:
    try:
      raw = input('You: ').strip()
    except (EOFError, KeyboardInterrupt):
      print()
      print('Bye.')
      break
    if not raw:
      continue
    if raw.lower() in {'q', 'quit', 'exit'}:
      print('Bye.')
      break
    print()
    print_result(run_turn(graph, raw, thread_id=thread_id))
    print()


if __name__ == '__main__':
  main()
