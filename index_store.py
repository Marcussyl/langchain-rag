#!/usr/bin/env python3
"""Shared Chroma index helpers for CLI RAG and LangGraph nodes."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Literal, TypedDict

import settings  # noqa: F401 — load .env before reading os.environ
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
UPLOAD_DIR = DATA_DIR / 'uploads'
CHROMA_ROOT = ROOT / '.chroma'
COLLECTION_PREFIX = 'tutor-notes'

ALLOWED_UPLOAD_SUFFIXES = {'.md', '.markdown', '.txt', '.pdf'}
MAX_UPLOAD_BYTES = int(os.getenv('RAG_MAX_UPLOAD_BYTES', str(5 * 1024 * 1024)))


OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')

SearchType = Literal['similarity', 'mmr']
ChatBackend = Literal['ollama', 'gemini', 'groq', 'openrouter']


def chat_backend() -> ChatBackend:
  """CHAT_BACKEND=ollama|gemini|groq|openrouter. Embeddings always stay on Ollama."""
  raw = (os.getenv('CHAT_BACKEND') or 'ollama').strip().lower()
  if raw in {'', 'ollama'}:
    return 'ollama'
  if raw in {'gemini', 'google'}:
    return 'gemini'
  if raw == 'groq':
    return 'groq'
  if raw in {'openrouter', 'or'}:
    return 'openrouter'
  raise ValueError(
    f'unsupported CHAT_BACKEND: {raw!r} (use ollama, gemini, groq, or openrouter)'
  )


def make_chat_model():
  """Chat LLM for RAG generate. Embeddings stay on Ollama regardless of backend."""
  backend = chat_backend()
  if backend == 'gemini':
    api_key = (os.getenv('GOOGLE_API_KEY') or '').strip()
    if not api_key:
      raise RuntimeError('CHAT_BACKEND=gemini requires GOOGLE_API_KEY in .env or the environment')
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = (os.getenv('GEMINI_MODEL') or 'gemini-2.0-flash').strip() or 'gemini-2.0-flash'
    print(f'(chat: gemini model={model_name})')
    # System / developer instructions require the Gemini Developer API v1beta path.
    return ChatGoogleGenerativeAI(
      model=model_name,
      temperature=0,
      google_api_key=api_key,
      api_version='v1beta',
    )

  if backend == 'groq':
    api_key = (os.getenv('GROQ_API_KEY') or '').strip()
    if not api_key:
      raise RuntimeError('CHAT_BACKEND=groq requires GROQ_API_KEY in .env or the environment')
    from langchain_groq import ChatGroq

    model_name = (os.getenv('GROQ_MODEL') or 'llama-3.1-8b-instant').strip() or 'llama-3.1-8b-instant'
    print(f'(chat: groq model={model_name})')
    return ChatGroq(model=model_name, temperature=0, api_key=api_key)

  if backend == 'openrouter':
    api_key = (os.getenv('OPENROUTER_API_KEY') or '').strip()
    if not api_key:
      raise RuntimeError(
        'CHAT_BACKEND=openrouter requires OPENROUTER_API_KEY in .env or the environment'
      )
    from langchain_openai import ChatOpenAI

    model_name = (
      os.getenv('OPENROUTER_MODEL') or 'meta-llama/llama-3.2-3b-instruct:free'
    ).strip() or 'meta-llama/llama-3.2-3b-instruct:free'
    print(f'(chat: openrouter model={model_name})')
    return ChatOpenAI(
      model=model_name,
      temperature=0,
      api_key=api_key,
      base_url='https://openrouter.ai/api/v1',
      default_headers={
        'HTTP-Referer': os.getenv('OPENROUTER_HTTP_REFERER', 'http://localhost'),
        'X-Title': os.getenv('OPENROUTER_APP_TITLE', 'langchain-rag'),
      },
    )

  print(f'(chat: ollama model={OLLAMA_MODEL})')
  return ChatOllama(model=OLLAMA_MODEL, temperature=0)


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


def _source_label(path: Path) -> str:
  """Prefer uploads/name.ext when file lives under data/uploads/."""
  try:
    rel = path.resolve().relative_to(DATA_DIR.resolve())
    return str(rel).replace('\\', '/')
  except ValueError:
    return path.name


def _iter_note_paths() -> list[Path]:
  DATA_DIR.mkdir(exist_ok=True)
  UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
  paths = sorted(
    {
      *DATA_DIR.glob('*.md'),
      *DATA_DIR.glob('*.txt'),
      *UPLOAD_DIR.glob('*.md'),
      *UPLOAD_DIR.glob('*.markdown'),
      *UPLOAD_DIR.glob('*.txt'),
      *UPLOAD_DIR.glob('*.pdf'),
    }
  )
  return paths


def _kind_for_path(path: Path) -> Literal['seed', 'upload']:
  try:
    path.resolve().relative_to(UPLOAD_DIR.resolve())
    return 'upload'
  except ValueError:
    return 'seed'


def list_note_files(*, config: RagConfig | None = None) -> list[dict]:
  """File-level corpus inventory for the Library UI (disk + optional chunk counts)."""
  cfg = config or DEFAULT_CONFIG
  chunk_counts: dict[str, int] = {}
  if cfg.persist_dir.exists():
    try:
      store = open_vectorstore(cfg)
      raw = store._collection.get(include=['metadatas'])
      for meta in raw.get('metadatas') or []:
        if not meta:
          continue
        source = str(meta.get('source') or '')
        if source:
          chunk_counts[source] = chunk_counts.get(source, 0) + 1
    except Exception:  # noqa: BLE001 — list still works if Chroma is unavailable
      chunk_counts = {}

  rows: list[dict] = []
  for path in _iter_note_paths():
    source = _source_label(path)
    stat = path.stat()
    rows.append(
      {
        'source': source,
        'filename': path.name,
        'kind': _kind_for_path(path),
        'suffix': path.suffix.lower(),
        'bytes': int(stat.st_size),
        'mtime': int(stat.st_mtime),
        'chunks': int(chunk_counts.get(source, 0)),
      }
    )
  return rows


def resolve_note_path(source: str) -> Path:
  """Map a source label (e.g. uploads/foo.md) to a path under data/."""
  cleaned = (source or '').replace('\\', '/').strip().lstrip('/')
  if not cleaned or cleaned in {'.', '..'} or '..' in cleaned.split('/'):
    raise ValueError('invalid source')
  path = (DATA_DIR / cleaned).resolve()
  data_root = DATA_DIR.resolve()
  if path != data_root and data_root not in path.parents:
    raise ValueError('source escapes data/')
  known = {p.resolve() for p in _iter_note_paths()}
  if path not in known:
    raise FileNotFoundError(f'unknown note: {cleaned}')
  return path


def delete_chroma_chunks_for_source(
  source: str,
  *,
  config: RagConfig | None = None,
) -> int:
  """Remove vectors whose metadata.source matches. Returns deleted id count."""
  cfg = config or DEFAULT_CONFIG
  if not cfg.persist_dir.exists():
    return 0
  store = open_vectorstore(cfg)
  result = store._collection.get(where={'source': source}, include=[])
  ids = list(result.get('ids') or [])
  if ids:
    store.delete(ids=ids)
  return len(ids)


def delete_note_file(source: str, *, config: RagConfig | None = None) -> dict:
  """Delete a note file under data/ and clear matching Chroma chunks."""
  path = resolve_note_path(source)
  label = _source_label(path)
  removed_chunks = delete_chroma_chunks_for_source(label, config=config)
  path.unlink(missing_ok=False)
  return {
    'source': label,
    'deleted_file': True,
    'removed_chunks': removed_chunks,
  }


def documents_from_file(path: Path) -> list[Document]:
  """Load one note file as LangChain Document(s)."""
  suffix = path.suffix.lower()
  source = _source_label(path)
  if suffix in {'.md', '.markdown', '.txt'}:
    text = path.read_text(encoding='utf-8')
    return [Document(page_content=text, metadata={'source': source})]
  if suffix == '.pdf':
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_index, page in enumerate(reader.pages, start=1):
      text = (page.extract_text() or '').strip()
      if not text:
        continue
      docs.append(
        Document(
          page_content=text,
          metadata={'source': source, 'page': page_index},
        )
      )
    if not docs:
      raise ValueError(f'no extractable text in PDF: {path.name}')
    return docs
  raise ValueError(f'unsupported file type: {suffix}')


def load_documents() -> list[Document]:
  """Load notes under data/ and data/uploads/ (creates a sample file if empty)."""
  paths = _iter_note_paths()
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
    docs.extend(documents_from_file(path))
  return docs


def safe_upload_name(filename: str) -> str:
  """Basename only; reject empty / odd names."""
  name = Path(filename or '').name.strip()
  if not name or name in {'.', '..'}:
    raise ValueError('missing filename')
  suffix = Path(name).suffix.lower()
  if suffix not in ALLOWED_UPLOAD_SUFFIXES:
    allowed = ', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))
    raise ValueError(f'unsupported type {suffix or "(none)"}; allowed: {allowed}')
  # Keep it filesystem-friendly on Windows.
  cleaned = ''.join(ch if ch.isalnum() or ch in {'-', '_', '.', ' '} else '_' for ch in name)
  cleaned = cleaned.strip().replace(' ', '_')
  if not cleaned or cleaned.startswith('.'):
    raise ValueError('invalid filename')
  return cleaned


def save_upload_bytes(filename: str, data: bytes) -> Path:
  """Write bytes under data/uploads/ (overwrites same name)."""
  if len(data) > MAX_UPLOAD_BYTES:
    raise ValueError(f'file too large (max {MAX_UPLOAD_BYTES} bytes)')
  if not data:
    raise ValueError('empty file')
  UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
  dest = UPLOAD_DIR / safe_upload_name(filename)
  dest.write_bytes(data)
  return dest


def ingest_file(path: Path, *, config: RagConfig | None = None, quiet: bool = False) -> dict:
  """Chunk + embed one file into the active Chroma collection (incremental).

  Replaces any existing chunks for the same source label so re-uploads do not
  leave duplicate vectors.
  """
  cfg = config or DEFAULT_CONFIG
  docs = documents_from_file(path)
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.chunk_size,
    chunk_overlap=cfg.chunk_overlap,
  )
  splits = splitter.split_documents(docs)
  source = _source_label(path)
  # Ensure collection exists, then replace prior vectors for this source.
  get_vectorstore(rebuild=False, quiet=True, config=cfg)
  removed = delete_chroma_chunks_for_source(source, config=cfg)
  store = open_vectorstore(cfg)
  store.add_documents(splits)
  if not quiet:
    print(
      f'(indexed upload {path.name}: {len(splits)} chunk(s)'
      f'{f", replaced {removed}" if removed else ""} | {cfg.label})'
    )
  return {
    'source': source,
    'path': str(path),
    'chunks': len(splits),
    'docs': len(docs),
    'replaced_chunks': removed,
  }


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


def _message_text(content: object) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts: list[str] = []
    for block in content:
      if isinstance(block, str):
        parts.append(block)
      elif isinstance(block, dict) and block.get('type') == 'text':
        parts.append(str(block.get('text') or ''))
      else:
        text = getattr(block, 'text', None)
        parts.append(text if isinstance(text, str) else str(block))
    return ''.join(parts)
  return str(content)


def generate_cited_answer(
  question: str,
  context: str,
  citations: list[Citation],
  *,
  on_token: Callable[[str], None] | None = None,
) -> str:
  """LLM body + fixed Sources footer.

  Optional on_token(str) receives incremental text chunks for SSE UIs.
  """
  prompt = ChatPromptTemplate.from_messages(
    [
      ('system', CITATION_SYSTEM),
      ('human', '{question}'),
    ]
  )
  model = make_chat_model()
  inputs = {
    'context': context or '(no context retrieved)',
    'question': question,
  }
  chain = prompt | model
  if on_token is None:
    result = chain.invoke(inputs)
    body = _message_text(getattr(result, 'content', result))
    return append_sources_footer(body, citations)

  parts: list[str] = []
  for chunk in chain.stream(inputs):
    text = _message_text(getattr(chunk, 'content', chunk))
    if not text:
      continue
    parts.append(text)
    on_token(text)
  body = ''.join(parts)
  return append_sources_footer(body, citations)
