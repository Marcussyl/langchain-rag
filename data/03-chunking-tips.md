# Chunking tips

`RecursiveCharacterTextSplitter` breaks long documents into overlapping pieces.

## Knobs in `rag.py`

- `chunk_size=400` — approximate characters per chunk
- `chunk_overlap=80` — overlap so sentences are not cut awkwardly

## Practical guidance

- Too large → retrieval returns noisy context; prompts get expensive.
- Too small → answers miss surrounding meaning.
- Prefer splitting on headings / paragraphs when documents are structured.
- Overlap helps when a key sentence sits on a boundary.

## Retriever `k`

`search_kwargs={'k': 3}` means “return the top 3 chunks”. Raise `k` if answers
feel incomplete; lower it if the model gets distracted by irrelevant text.
