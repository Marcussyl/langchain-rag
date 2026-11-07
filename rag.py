#!/usr/bin/env python3
"""Minimal local RAG: load → split → embed (Ollama) → Chroma → answer.

Prereqs:
  ollama serve
  ollama pull llama3.2
  ollama pull nomic-embed-text

Run from repo root:
  uv run python rag.py
"""

from __future__ import annotations

import os
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
  """Load every .md / .txt under data/ (creates sample files if empty)."""
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


def build_vectorstore(docs: list[Document]) -> Chroma:
  splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
  splits = splitter.split_documents(docs)
  embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
  return Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=str(CHROMA_DIR),
    collection_name=COLLECTION,
  )


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


def main() -> None:
  docs = load_documents()
  print(f'(loaded {len(docs)} document(s) from {DATA_DIR})')
  store = build_vectorstore(docs)
  query = os.getenv('RAG_QUERY', 'What is RAG and what vector store does this project use?')
  print(f'(query) {query}')
  print()
  print('Bot:', answer(query, store))


if __name__ == '__main__':
  main()
