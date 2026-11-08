#!/usr/bin/env python3
"""Compare retrieval quality across chunk_size / k / search_type (Sources only).

Does not call the chat LLM by default — focus on which chunks land in Sources.

  uv run python compare_retrieve.py -q 'What is RAG?'
  uv run python compare_retrieve.py -q 'What is RAG?' --answer
  uv run python compare_retrieve.py -q '...' --rebuild
"""

from __future__ import annotations

import argparse

from index_store import (
  RagConfig,
  append_sources_footer,
  generate_cited_answer,
  get_vectorstore,
  retrieve_context,
)

# Presets: baseline vs larger chunks vs higher k vs MMR (same chunking as baseline).
PRESETS: list[RagConfig] = [
  RagConfig(chunk_size=400, chunk_overlap=80, k=3, search_type='similarity'),
  RagConfig(chunk_size=800, chunk_overlap=160, k=3, search_type='similarity'),
  RagConfig(chunk_size=400, chunk_overlap=80, k=5, search_type='similarity'),
  RagConfig(chunk_size=400, chunk_overlap=80, k=3, search_type='mmr'),
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Compare RAG retrieval presets')
  parser.add_argument('-q', '--query', required=True, help='Question to retrieve for')
  parser.add_argument(
    '--rebuild',
    action='store_true',
    help='Rebuild each preset index before comparing',
  )
  parser.add_argument(
    '--answer',
    action='store_true',
    help='Also run the LLM for each preset (slower)',
  )
  return parser.parse_args()


def run_preset(query: str, config: RagConfig, *, rebuild: bool, with_answer: bool) -> None:
  print('=' * 72)
  print(f'preset: {config.label}')
  print('=' * 72)
  store = get_vectorstore(rebuild=rebuild, config=config)
  context, citations, _hits = retrieve_context(query, store, config=config)
  # Always show the Sources block (empty body) so presets are easy to scan.
  print(append_sources_footer('(retrieval only)', citations))
  if with_answer:
    print()
    print('Bot:', generate_cited_answer(query, context, citations))
  print()


def _wipe_unique_indexes() -> None:
  """Delete each persist_dir once before any Chroma client opens (avoids WinError 32)."""
  import shutil

  seen: set[str] = set()
  for config in PRESETS:
    key = str(config.persist_dir)
    if key in seen:
      continue
    seen.add(key)
    if not config.persist_dir.exists():
      continue
    try:
      shutil.rmtree(config.persist_dir)
      print(f'(removed old index at {config.persist_dir})')
    except PermissionError:
      print(
        f'(warn: locked {config.persist_dir} — stop `langgraph.dev` / uvicorn, then retry --rebuild)'
      )


def main() -> None:
  args = parse_args()
  print(f'(query) {args.query}')
  print(f'(presets) {len(PRESETS)} — chunk_size / k / similarity|mmr')
  print()
  if args.rebuild:
    _wipe_unique_indexes()
  for config in PRESETS:
    # Indexes were wiped above when --rebuild; never rmtree after a client is open.
    run_preset(args.query, config, rebuild=False, with_answer=args.answer)


if __name__ == '__main__':
  main()
