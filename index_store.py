#!/usr/bin/env python3
"""Shared Chroma index helpers for CLI RAG and LangGraph nodes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from typing import TypedDict

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
RETRIEVE_K = int(os.getenv('RAG_RETRIEVE_K', '3'))


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


def get_vectorstore(*, rebuild: bool = False, quiet: bool = False) -> Chroma:
  if rebuild and CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)
    if not quiet:
      print('(removed old .chroma index)')

  if not rebuild and CHROMA_DIR.exists():
    if not quiet:
      print(f'(reusing index at {CHROMA_DIR})')
    return open_vectorstore()

  docs = load_documents()
  if not quiet:
    print(f'(indexing {len(docs)} document(s) from {DATA_DIR})')
  store = build_vectorstore(docs)
  if not quiet:
    print(f'(wrote index to {CHROMA_DIR})')
  return store


class Citation(TypedDict):
  index: int
  source: str
  preview: str


# Ask for inline [n] cites; Sources footer is appended in Python (deterministic).
CITATION_SYSTEM = (
  'Answer using ONLY the numbered context chunks below. '
  'If the context is not enough, say so.\n'
  'When you use a fact, cite it inline like [1] or [2] matching the chunk numbers.\n'
  'Do not invent a Sources section; that will be appended automatically.\n\n'
  'Context:\n{context}'
)


def _preview(text: str, limit: int = 80) -> str:
  compact = ' '.join(text.split())
  if len(compact) <= limit:
    return compact
  return compact[: limit - 1] + '…'


def format_hits(hits: list[Document]) -> tuple[str, list[Citation]]:
  """Build numbered prompt context and citation rows for a fixed Sources footer."""
  citations: list[Citation] = []
  parts: list[str] = []
  for index, doc in enumerate(hits, start=1):
    source = str(doc.metadata.get('source', '?'))
    citations.append(
      {
        'index': index,
        'source': source,
        'preview': _preview(doc.page_content),
      }
    )
    parts.append(f'[{index}] source: {source}\n{doc.page_content}')
  return '\n\n'.join(parts), citations


def citation_sources(citations: list[Citation]) -> list[str]:
  """Unique filenames in first-seen order (for CLI / state)."""
  sources: list[str] = []
  for row in citations:
    if row['source'] not in sources:
      sources.append(row['source'])
  return sources


def append_sources_footer(answer: str, citations: list[Citation]) -> str:
  """Append a deterministic Sources block (do not trust the LLM for this)."""
  body = (answer or '').rstrip()
  if not citations:
    return f'{body}\n\nSources: (none retrieved)'
  lines = [f'{row["index"]}. {row["source"]} — {row["preview"]}' for row in citations]
  return f'{body}\n\nSources:\n' + '\n'.join(lines)


def retrieve_context(
  question: str,
  vectorstore: Chroma | None = None,
) -> tuple[str, list[Citation], list[Document]]:
  store = vectorstore or get_vectorstore(rebuild=False, quiet=True)
  hits = store.as_retriever(search_kwargs={'k': RETRIEVE_K}).invoke(question)
  context, citations = format_hits(hits)
  return context, citations, hits


def generate_cited_answer(question: str, context: str, citations: list[Citation]) -> str:
  """LLM body + fixed Sources footer."""
  prompt = ChatPromptTemplate.from_messages(
    [
      ('system', CITATION_SYSTEM),
      ('human', '{question}'),
    ]
  )
  model = ChatOllama(model=OLLAMA_MODEL, temperature=0)
  result = (prompt | model).invoke(
    {
      'context': context or '(no context retrieved)',
      'question': question,
    }
  )
  content = getattr(result, 'content', result)
  body = content if isinstance(content, str) else str(content)
  return append_sources_footer(body, citations)
