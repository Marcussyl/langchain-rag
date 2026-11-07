#!/usr/bin/env python3
"""Local thread persistence for CLI / FastAPI (not for langgraph.dev export).

Studio rejects graphs compiled with SqliteSaver — keep graph_server.py checkpointer-free.
Postgres (production): install langgraph-checkpoint-postgres and swap savers later.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / 'checkpoints.sqlite'


def make_sqlite_checkpointer(path: Path | None = None) -> SqliteSaver:
  db = path or CHECKPOINT_PATH
  conn = sqlite3.connect(str(db), check_same_thread=False)
  checkpointer = SqliteSaver(conn)
  checkpointer.setup()
  return checkpointer
