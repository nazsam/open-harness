"""Long-term memory: facts an agent keeps across sessions.

Sessions hold the conversation. Memory holds durable facts ("the user prefers
metric units") that should outlive any single conversation. Give an agent a
memory store and it gets two tools, ``remember`` and ``recall``, and the most
relevant memories are added to its instructions at the start of every run.

The built-in stores rank by keyword overlap (BM25). Implement the ``Memory``
protocol to plug in a vector database.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .types import new_id

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have i in is it its of on or that the this to was were "
    "will with you your my me we our what who when where how do does did".split()
)


def _stem(word: str) -> str:
    """Very light stemming so "likes"/"liked"/"liking" match "like"."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


@dataclass
class MemoryItem:
    text: str
    id: str = field(default_factory=lambda: new_id("mem"))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    score: float = 0.0


@runtime_checkable
class Memory(Protocol):
    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryItem: ...
    async def search(self, query: str, limit: int = 5) -> list[MemoryItem]: ...
    async def list(self) -> list[MemoryItem]: ...
    async def delete(self, item_id: str) -> bool: ...
    async def clear(self) -> None: ...


def rank(items: list[MemoryItem], query: str, limit: int) -> list[MemoryItem]:
    """BM25 ranking over item text. Returns the newest items when the query has no keywords."""
    q = tokenize(query)
    if not q:
        return sorted(items, key=lambda i: i.created_at, reverse=True)[:limit]
    docs = [tokenize(i.text) for i in items]
    n = len(docs) or 1
    avg = sum(len(d) for d in docs) / n or 1
    df: dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    scored = []
    for item, d in zip(items, docs, strict=True):
        s = 0.0
        for w in q:
            tf = d.count(w)
            if not tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(d) / avg))
        if s > 0:
            scored.append(MemoryItem(item.text, item.id, item.metadata, item.created_at, round(s, 4)))
    scored.sort(key=lambda i: (i.score, i.created_at), reverse=True)
    return scored[:limit]


class InMemoryMemory:
    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryItem:
        for existing in self._items:  # skip exact duplicates
            if existing.text.strip().lower() == text.strip().lower():
                return existing
        item = MemoryItem(text=text.strip(), metadata=metadata or {})
        self._items.append(item)
        return item

    async def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        return rank(self._items, query, limit)

    async def list(self) -> list[MemoryItem]:
        return list(self._items)

    async def delete(self, item_id: str) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if i.id != item_id]
        return len(self._items) < before

    async def clear(self) -> None:
        self._items.clear()


class FileMemory(InMemoryMemory):
    """Memory persisted to a JSON file."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path).expanduser()
        self._lock = asyncio.Lock()
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8") or "[]")
            self._items = [MemoryItem(**{k: v for k, v in d.items() if k != "score"}) for d in data]

    async def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            [{k: v for k, v in asdict(i).items() if k != "score"} for i in self._items], indent=2, ensure_ascii=False
        )
        await asyncio.to_thread(self.path.write_text, payload, encoding="utf-8")

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryItem:
        async with self._lock:
            item = await super().add(text, metadata)
            await self._save()
            return item

    async def delete(self, item_id: str) -> bool:
        async with self._lock:
            ok = await super().delete(item_id)
            await self._save()
            return ok

    async def clear(self) -> None:
        async with self._lock:
            await super().clear()
            await self._save()


def memory_tools(memory: Memory) -> list[Any]:
    """The ``remember`` and ``recall`` tools given to agents that have memory."""
    from .tools import tool

    @tool
    async def remember(fact: str) -> str:
        """Save a durable fact about the user or task for future conversations.

        Only store stable, useful facts (preferences, names, decisions), never secrets.

        Args:
            fact: One short, self-contained sentence.
        """
        item = await memory.add(fact)
        return f"Saved memory {item.id}"

    @tool
    async def recall(query: str) -> list[str]:
        """Search long-term memory for facts related to a query.

        Args:
            query: What to look for.
        """
        return [i.text for i in await memory.search(query, limit=5)]

    return [remember, recall]
