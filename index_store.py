#!/usr/bin/env python3
"""Shared Chroma index helpers for CLI RAG and LangGraph nodes."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
CHROMA_ROOT = ROOT / '.chroma'
COLLECTION_PREFIX = 'tutor-notes'

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')

SearchType = Literal['similarity', 'mmr']


@dataclass(frozen=True)
class RagConfig:
  """Retrieval / chunking knobs used by index + compare scripts."""

  chunk_size: int = 400
  chunk_overlap: int = 80
  k: int = 3
  search_type: SearchType = 'similarity'

  def __post_init__(self) -> None:
    if self.search_type not in {'similarity', 'mmr'}:
      raise ValueError(f'unsupported search_type: {self.search_type}')
    if self.chunk_overlap >= self.chunk_size:
      raise ValueError('chunk_overlap must be smaller than chunk_size')

  @classmethod
  def from_env(cls) -> RagConfig:
    search = os.getenv('RAG_SEARCH_TYPE', 'similarity')
    if search not in {'similarity', 'mmr'}:
      raise ValueError(f'unsupported RAG_SEARCH_TYPE: {search}')
    return cls(
      chunk_size=int(os.getenv('RAG_CHUNK_SIZE', '400')),
      chunk_overlap=int(os.getenv('RAG_CHUNK_OVERLAP', '80')),
      k=int(os.getenv('RAG_RETRIEVE_K', '3')),
      search_type=search,  # type: ignore[arg-type]
    )

  @property
  def label(self) -> str:
    return (
      f'chunk={self.chunk_size}/overlap={self.chunk_overlap}/'
      f'k={self.k}/search={self.search_type}'
    )

  @property
  def persist_dir(self) -> Path:
    # Separate dirs so compare presets do not clobber each other.
    return CHROMA_ROOT / f'c{self.chunk_size}_o{self.chunk_overlap}'

  @property
  def collection_name(self) -> str:
    return f'{COLLECTION_PREFIX}-c{self.chunk_size}-o{self.chunk_overlap}'


# Back-compat names used by older imports / docs.
DEFAULT_CONFIG = RagConfig.from_env()
RETRIEVE_K = DEFAULT_CONFIG.k
CHROMA_DIR = DEFAULT_CONFIG.persist_dir
COLLECTION = DEFAULT_CONFIG.collection_name


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


def build_vectorstore(docs: list[Document], config: RagConfig | None = None) -> Chroma:
  cfg = config or DEFAULT_CONFIG
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.chunk_size,
    chunk_overlap=cfg.chunk_overlap,
  )
  splits = splitter.split_documents(docs)
  cfg.persist_dir.mkdir(parents=True, exist_ok=True)
  return Chroma.from_documents(
    documents=splits,
    embedding=_embeddings(),
    persist_directory=str(cfg.persist_dir),
    collection_name=cfg.collection_name,
  )


def open_vectorstore(config: RagConfig | None = None) -> Chroma:
  """Open an existing on-disk Chroma collection for this config."""
  cfg = config or DEFAULT_CONFIG
  return Chroma(
    persist_directory=str(cfg.persist_dir),
    embedding_function=_embeddings(),
    collection_name=cfg.collection_name,
  )


def get_vectorstore(
  *,
  rebuild: bool = False,
  quiet: bool = False,
  config: RagConfig | None = None,
) -> Chroma:
  cfg = config or DEFAULT_CONFIG
  if rebuild and cfg.persist_dir.exists():
    shutil.rmtree(cfg.persist_dir)
    if not quiet:
      print(f'(removed old index at {cfg.persist_dir})')

  if not rebuild and cfg.persist_dir.exists():
    if not quiet:
      print(f'(reusing index at {cfg.persist_dir} | {cfg.label})')
    return open_vectorstore(cfg)

  docs = load_documents()
  if not quiet:
    print(f'(indexing {len(docs)} document(s) from {DATA_DIR} | {cfg.label})')
  store = build_vectorstore(docs, cfg)
  if not quiet:
    print(f'(wrote index to {cfg.persist_dir})')
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


def make_retriever(store: Chroma, config: RagConfig | None = None):
  cfg = config or DEFAULT_CONFIG
  if cfg.search_type == 'mmr':
    return store.as_retriever(
      search_type='mmr',
      search_kwargs={
        'k': cfg.k,
        'fetch_k': max(20, cfg.k * 4),
        'lambda_mult': 0.5,
      },
    )
  return store.as_retriever(search_kwargs={'k': cfg.k})


def retrieve_context(
  question: str,
  vectorstore: Chroma | None = None,
  config: RagConfig | None = None,
) -> tuple[str, list[Citation], list[Document]]:
  cfg = config or DEFAULT_CONFIG
  store = vectorstore or get_vectorstore(rebuild=False, quiet=True, config=cfg)
  hits = make_retriever(store, cfg).invoke(question)
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
