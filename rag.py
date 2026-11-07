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
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
CHROMA_DIR = ROOT / '.chroma'
COLLECTION = 'tutor-notes'

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')


def load_documents() -> list[Document]:
  """Load every .md / .txt under data/ (creates a sample file if empty)."""
  DATA_DIR.mkdir(exist_ok=True)
  paths = sorted([*DATA_DIR.glob('*.md'), *DATA_DIR.glob('*.txt')])
  if not paths:
    sample = DATA_DIR / 'sample.md'
    sample.write_text(
      '# Sample notes\n\n'
      'RAG means Retrieval Augmented Generation: fetch relevant chunks, then ask the LLM.\n\n'
      'This project uses local Ollama embeddings and a Chroma vector store on disk.\n',
      encoding='utf-8',
    )
    paths = [sample]

  docs: list[Document] = []
  for path in paths:
    text = path.read_text(encoding='utf-8')
    docs.append(Document(page_content=text, metadata={'source': str(path.name)}))
  return docs


def _embeddings() -> OllamaEmbeddings:
  return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)


def build_vectorstore(docs: list[Document]) -> Chroma:
  splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
  splits = splitter.split_documents(docs)
  return Chroma.from_documents(
    documents=splits,
    embedding=_embeddings(),
    persist_directory=str(CHROMA_DIR),
    collection_name=COLLECTION,
  )


def open_vectorstore() -> Chroma:
  """Open an existing on-disk Chroma collection."""
  return Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=_embeddings(),
    collection_name=COLLECTION,
  )


def get_vectorstore(*, rebuild: bool) -> Chroma:
  if rebuild and CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)
    print('(removed old .chroma index)')

  if not rebuild and CHROMA_DIR.exists():
    print(f'(reusing index at {CHROMA_DIR})')
    return open_vectorstore()

  docs = load_documents()
  print(f'(indexing {len(docs)} document(s) from {DATA_DIR})')
  store = build_vectorstore(docs)
  print(f'(wrote index to {CHROMA_DIR})')
  return store


def answer(query: str, vectorstore: Chroma) -> str:
  retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
  hits = retriever.invoke(query)
  context = '\n\n'.join(
    f'[{doc.metadata.get("source", "?")}]\n{doc.page_content}' for doc in hits
  )
  print('(retrieved chunks)')
  for index, doc in enumerate(hits, start=1):
    preview = doc.page_content.replace('\n', ' ')[:120]
    print(f'  {index}. {doc.metadata.get("source")}: {preview}…')

  prompt = ChatPromptTemplate.from_messages(
    [
      (
        'system',
        'Answer using ONLY the context below. If the context is not enough, say so.\n\n'
        'Context:\n{context}',
      ),
      ('human', '{question}'),
    ]
  )
  model = ChatOllama(model=OLLAMA_MODEL, temperature=0)
  chain = prompt | model
  result = chain.invoke({'context': context, 'question': query})
  content = getattr(result, 'content', result)
  return content if isinstance(content, str) else str(content)


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
    print(f'(note: RAG_QUERY is set in the shell but ignored; use -q to one-shot, or clear it)')

  print('Interactive RAG. Empty input ignored; type q to quit.')
  print('After editing files in data/, restart with: uv run python rag.py --rebuild')
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
