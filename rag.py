#!/usr/bin/env python3
"""Minimal local RAG: load → split → embed (Ollama) → Chroma → answer.

Prereqs:
  ollama serve
  ollama pull llama3.2
  ollama pull nomic-embed-text

Run from repo root:
  uv run python rag.py              # interactive Q&A (reuses .chroma if present)
  uv run python rag.py --rebuild    # re-index data/ then chat
  uv run python rag.py -q '...'     # one-shot question

LangGraph version (retrieve → generate nodes + Studio):
  uv run python graph_chat.py
  uv run langgraph dev
"""

from __future__ import annotations

import argparse
import os

from langchain_chroma import Chroma

from index_store import generate_cited_answer, get_vectorstore, retrieve_context


def answer(query: str, vectorstore: Chroma) -> str:
  context, citations, hits = retrieve_context(query, vectorstore)
  print('(retrieved chunks)')
  for row, doc in zip(citations, hits, strict=True):
    print(f'  [{row["index"]}] {row["source"]}: {row["preview"]}')
  return generate_cited_answer(query, context, citations)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Local Ollama + Chroma RAG demo')
  parser.add_argument(
    '-q',
    '--query',
    default=None,
    help='One-shot question (skip interactive loop). Do not rely on leftover RAG_QUERY env.',
  )
  parser.add_argument(
    '--rebuild',
    action='store_true',
    help='Delete .chroma and re-index everything under data/',
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  store = get_vectorstore(rebuild=args.rebuild)

  # Only -q/--query triggers one-shot. A leftover $env:RAG_QUERY must not steal the REPL.
  if args.query:
    print(f'(query) {args.query}')
    print()
    print('Bot:', answer(args.query, store))
    return

  leftover = os.getenv('RAG_QUERY')
  if leftover:
    print('(note: RAG_QUERY is set in the shell but ignored; use -q to one-shot, or clear it)')

  print('Interactive RAG. Empty input ignored; type q to quit.')
  print('After editing files in data/, restart with: uv run python rag.py --rebuild')
  print('LangGraph + Studio: uv run python graph_chat.py  |  uv run langgraph dev')
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
    print('Bot:', answer(raw, store))
    print()


if __name__ == '__main__':
  main()
