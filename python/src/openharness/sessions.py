"""Sessions keep conversation history between runs.

Pass a session to ``run`` and the agent remembers earlier turns:

    session = FileSession("chats/alice.jsonl")
    await run(agent, "My name is Alice", session=session)
    await run(agent, "What is my name?", session=session)  # -> "Alice"

Implement the ``Session`` protocol to store history anywhere (Redis,
Postgres, your own API).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

from .types import Message


@runtime_checkable
class Session(Protocol):
    async def get_messages(self, limit: int | None = None) -> list[Message]: ...
    async def add_messages(self, messages: list[Message]) -> None: ...
    async def clear(self) -> None: ...


def _trim(messages: list[Message], limit: int | None) -> list[Message]:
    """Keep the last ``limit`` messages without splitting a tool call from its results."""
    if limit is None or len(messages) <= limit:
        return list(messages)
    out = messages[-limit:]
    while out and out[0].role == "tool":
        out = out[1:]
    return out


class InMemorySession:
    """History held in memory for the life of the object."""

    def __init__(self, max_messages: int | None = None):
        self._messages: list[Message] = []
        self.max_messages = max_messages

    async def get_messages(self, limit: int | None = None) -> list[Message]:
        return _trim(self._messages, limit or self.max_messages)

    async def add_messages(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    async def clear(self) -> None:
        self._messages.clear()


class FileSession:
    """History appended to a JSON Lines file. Human readable and easy to back up."""

    def __init__(self, path: str | Path, max_messages: int | None = None):
        self.path = Path(path).expanduser()
        self.max_messages = max_messages
        self._lock = asyncio.Lock()

    async def get_messages(self, limit: int | None = None) -> list[Message]:
        if not self.path.exists():
            return []
        text = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        msgs = [Message.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]
        return _trim(msgs, limit or self.max_messages)

    async def add_messages(self, messages: list[Message]) -> None:
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lines = "".join(json.dumps(m.to_dict(), ensure_ascii=False) + "\n" for m in messages)

            def write() -> None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(lines)

            await asyncio.to_thread(write)

    async def clear(self) -> None:
        async with self._lock:
            if self.path.exists():
                await asyncio.to_thread(self.path.unlink)


class SQLiteSession:
    """History stored in SQLite. One database can hold many sessions."""

    def __init__(self, session_id: str, db_path: str | Path = "openharness.db", max_messages: int | None = None):
        self.session_id = session_id
        self.db_path = str(Path(db_path).expanduser()) if str(db_path) != ":memory:" else ":memory:"
        self.max_messages = max_messages
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, data TEXT NOT NULL, created_at REAL DEFAULT (julianday('now')))"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")
        self._conn.commit()
        self._lock = asyncio.Lock()

    async def get_messages(self, limit: int | None = None) -> list[Message]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM messages WHERE session_id=? ORDER BY id", (self.session_id,)
            ).fetchall()
        return _trim([Message.from_dict(json.loads(r[0])) for r in rows], limit or self.max_messages)

    async def add_messages(self, messages: list[Message]) -> None:
        async with self._lock:
            self._conn.executemany(
                "INSERT INTO messages (session_id, data) VALUES (?, ?)",
                [(self.session_id, json.dumps(m.to_dict(), ensure_ascii=False)) for m in messages],
            )
            self._conn.commit()

    async def clear(self) -> None:
        async with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id=?", (self.session_id,))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
