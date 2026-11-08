#!/usr/bin/env python3
"""Thread persistence for CLI / FastAPI (not for langgraph.dev export).

Default: SqliteSaver → checkpoints.sqlite
Optional: set DATABASE_URL → PostgresSaver (see docker-compose.yml)

Studio rejects graphs compiled with a custom checkpointer — keep
graph_server.py checkpointer-free.
"""

from __future__ import annotations

import atexit
import os
import sqlite3
from pathlib import Path
from typing import Any

import settings  # noqa: F401 — load .env before reading DATABASE_URL
from langgraph.checkpoint.sqlite import SqliteSaver

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / 'checkpoints.sqlite'

# Keep pool/connection alive for the process lifetime (FastAPI / long CLI).
_POOL: Any = None
_SQLITE_CONN: sqlite3.Connection | None = None
_ATEXIT_REGISTERED = False


def database_url() -> str | None:
  url = (os.getenv('DATABASE_URL') or '').strip()
  return url or None


def close_checkpointer_resources() -> None:
  """Close Postgres pool / Sqlite connection (avoids CLI shutdown warnings)."""
  global _POOL, _SQLITE_CONN
  if _POOL is not None:
    try:
      _POOL.close()
    except Exception:
      pass
    _POOL = None
  if _SQLITE_CONN is not None:
    try:
      _SQLITE_CONN.close()
    except Exception:
      pass
    _SQLITE_CONN = None


def _ensure_atexit() -> None:
  global _ATEXIT_REGISTERED
  if not _ATEXIT_REGISTERED:
    atexit.register(close_checkpointer_resources)
    _ATEXIT_REGISTERED = True


def make_sqlite_checkpointer(path: Path | None = None) -> SqliteSaver:
  global _SQLITE_CONN
  _ensure_atexit()
  db = path or CHECKPOINT_PATH
  if _SQLITE_CONN is None:
    _SQLITE_CONN = sqlite3.connect(str(db), check_same_thread=False)
  checkpointer = SqliteSaver(_SQLITE_CONN)
  checkpointer.setup()
  return checkpointer


def make_postgres_checkpointer(conninfo: str):
  """Long-lived PostgresSaver backed by a connection pool."""
  global _POOL
  _ensure_atexit()
  from langgraph.checkpoint.postgres import PostgresSaver
  from psycopg.rows import dict_row
  from psycopg_pool import ConnectionPool

  if _POOL is None:
    _POOL = ConnectionPool(
      conninfo=conninfo,
      max_size=5,
      kwargs={
        'autocommit': True,
        'prepare_threshold': 0,
        'row_factory': dict_row,
      },
    )
  checkpointer = PostgresSaver(_POOL)
  checkpointer.setup()
  return checkpointer


def make_checkpointer(*, quiet: bool = False):
  """Sqlite by default; Postgres when DATABASE_URL is set."""
  url = database_url()
  if url:
    if not quiet:
      print('(checkpointer: postgres via DATABASE_URL)')
    return make_postgres_checkpointer(url)
  if not quiet:
    print(f'(checkpointer: sqlite at {CHECKPOINT_PATH})')
  return make_sqlite_checkpointer()
