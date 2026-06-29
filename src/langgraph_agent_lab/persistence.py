"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _sqlite_path(database_url: str | None) -> Path:
    if not database_url:
        return Path("outputs/langgraph_checkpoints.sqlite")
    if database_url.startswith("sqlite:///"):
        return Path(database_url.removeprefix("sqlite:///"))
    return Path(database_url)


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer."""
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError("Install: pip install langgraph-checkpoint-sqlite") from exc

        path = _sqlite_path(database_url)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        saver = SqliteSaver(conn=conn)
        if hasattr(saver, "setup"):
            saver.setup()
        return saver
    if kind == "postgres":
        raise NotImplementedError("Postgres checkpointer is an optional extension")
    raise ValueError(f"Unknown checkpointer kind: {kind}")
