#!/usr/bin/env python3
"""Load repo-root .env into os.environ for CLI / API (Studio also uses langgraph.json env)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path = ROOT / '.env') -> None:
  """Load KEY=VALUE lines. Skip empty values so code defaults still apply.

  Does not override variables already set in the process environment.
  """
  if not path.is_file():
    return
  for raw in path.read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if not line or line.startswith('#') or '=' not in line:
      continue
    key, _, value = line.partition('=')
    key = key.strip()
    value = value.strip().strip("'").strip('"')
    if not key or key in os.environ:
      continue
    if value == '':
      # Leave unset so os.getenv(name, default) keeps working.
      continue
    os.environ[key] = value


load_dotenv()
