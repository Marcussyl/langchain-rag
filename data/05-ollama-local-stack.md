# Local Ollama stack checklist

## Services

1. Start Ollama (`ollama serve` if it is not already running).
2. Pull chat and embed models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Run RAG

```bash
uv sync
uv run python rag.py
```

Optional query override (PowerShell):

```powershell
$env:RAG_QUERY = 'How does chunk overlap help?'
uv run python rag.py
```

## Common failures

- Embed model missing → Ollama 404 on `nomic-embed-text`.
- Chat model missing → Ollama 404 on `llama3.2`.
- Empty `data/` → `rag.py` writes a tiny fallback sample; prefer real notes.
