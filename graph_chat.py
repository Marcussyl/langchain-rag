#!/usr/bin/env python3
"""CLI for the LangGraph RAG pipeline (retrieve → generate).

Run from repo root:
  uv run python graph_chat.py
  uv run python graph_chat.py --rebuild
  uv run python graph_chat.py -q 'What is RAG?'

Studio (same graph):
  uv run langgraph dev
"""

from __future__ import annotations

import argparse
import os

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
    help='Delete .chroma and re-index everything under data/',
  )
  return parser.parse_args()


def print_result(result: dict) -> None:
  # Full answer already includes the deterministic Sources footer.
  print('Bot:', result.get('answer') or '(no answer)')


def main() -> None:
  args = parse_args()
  get_vectorstore(rebuild=args.rebuild)
  graph = build_graph(ensure_index=False)

  if args.query:
    print(f'(query) {args.query}')
    print()
    print_result(run_turn(graph, args.query))
    return

  leftover = os.getenv('RAG_QUERY')
  if leftover:
    print('(note: RAG_QUERY is set in the shell but ignored; use -q to one-shot, or clear it)')

  print('Interactive LangGraph RAG (retrieve → generate). Empty input ignored; type q to quit.')
  print('Studio: uv run langgraph dev  (select graph: rag)')
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
    print_result(run_turn(graph, raw))
    print()


if __name__ == '__main__':
  main()
