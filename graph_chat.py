#!/usr/bin/env python3
"""CLI for the LangGraph RAG pipeline (retrieve → generate + Sqlite threads).

Run from repo root:
  uv run python graph_chat.py
  uv run python graph_chat.py --thread demo
  uv run python graph_chat.py --thread demo --checkpoint <id> -q '...'

Interactive: type `history` to list checkpoint_ids for this thread.
Studio (platform checkpointer): uv run langgraph dev
"""

from __future__ import annotations

import argparse
import os

from checkpoint_store import make_checkpointer
from graph_rag import (
  build_graph,
  latest_checkpoint_id,
  print_checkpoint_history,
  run_turn,
)
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
  parser.add_argument(
    '--checkpoint',
    default=None,
    help='Optional checkpoint_id to fork/continue from (see interactive `history`).',
  )
  return parser.parse_args()


def print_result(result: dict, *, thread_id: str, graph) -> None:
  print('Bot:', result.get('answer') or '(no answer)')
  messages = result.get('messages') or []
  ckpt = latest_checkpoint_id(graph, thread_id)
  print(f'(thread messages: {len(messages)}; checkpoint_id={ckpt})')


def main() -> None:
  args = parse_args()
  get_vectorstore(rebuild=args.rebuild)
  checkpointer = make_checkpointer()
  graph = build_graph(ensure_index=False, checkpointer=checkpointer)
  thread_id = args.thread.strip() or 'default'
  checkpoint_id = (args.checkpoint or '').strip() or None

  if args.query:
    print(f'(query) {args.query}')
    print(f'(thread) {thread_id}')
    if checkpoint_id:
      print(f'(from checkpoint) {checkpoint_id}')
    print()
    print_result(
      run_turn(graph, args.query, thread_id=thread_id, checkpoint_id=checkpoint_id),
      thread_id=thread_id,
      graph=graph,
    )
    return

  leftover = os.getenv('RAG_QUERY')
  if leftover:
    print('(note: RAG_QUERY is set in the shell but ignored; use -q to one-shot, or clear it)')

  print('Interactive LangGraph RAG. Empty input ignored; type q to quit.')
  print(f'thread_id={thread_id}. Commands: history | use <checkpoint_id> | clearckpt')
  print('Studio: uv run langgraph dev')
  print()
  active_checkpoint: str | None = checkpoint_id
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
    if raw.lower() == 'history':
      print_checkpoint_history(graph, thread_id)
      continue
    if raw.lower() == 'clearckpt':
      active_checkpoint = None
      print('(cleared checkpoint_id; next turn uses thread tip)')
      continue
    if raw.lower().startswith('use '):
      active_checkpoint = raw[4:].strip() or None
      print(f'(next turn from checkpoint_id={active_checkpoint})')
      continue
    print()
    print_result(
      run_turn(
        graph,
        raw,
        thread_id=thread_id,
        checkpoint_id=active_checkpoint,
      ),
      thread_id=thread_id,
      graph=graph,
    )
    # After a successful fork turn, follow the new tip unless user pins again.
    active_checkpoint = None
    print()


if __name__ == '__main__':
  main()
