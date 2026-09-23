---
title: Sessions and memory
parent: Guides
nav_order: 5
---

# Sessions and memory
{: .no_toc }

1. TOC
{:toc}

**Sessions** hold one conversation. **Memory** holds durable facts that should outlive any single conversation.

## Sessions

Pass a session to `run` and the agent sees earlier turns. New messages are saved when the run succeeds.

```python
from openharness import FileSession, InMemorySession, SQLiteSession

session = SQLiteSession("user-42", "data/chats.db")      # or FileSession("chats/42.jsonl"), InMemorySession()
await run(agent, "My name is Alice", session=session)
await run(agent, "What's my name?", session=session)     # "Alice"
```

```ts
import { FileSession, InMemorySession } from "openharness";

const session = new FileSession("chats/42.jsonl");
await run(agent, "My name is Alice", { session });
await run(agent, "What's my name?", { session });
```

| Store | Python | TypeScript |
|---|---|---|
| Memory | `InMemorySession()` | `new InMemorySession()` |
| JSON Lines file | `FileSession(path)` | `new FileSession(path)` |
| SQLite | `SQLiteSession(id, db_path)` | (use your own) |

The JSON Lines format is identical in both SDKs, so a Python service and a Node service can share a conversation file.

Keep long conversations bounded with `max_messages` (`maxMessages`): only the most recent messages are sent, and a tool
result is never separated from the call that produced it.

### Your own store

Implement three methods: `get_messages(limit)`, `add_messages(messages)` and `clear()`
(`getMessages`, `addMessages`, `clear`). Messages serialise with `to_dict()` / `messageToJSON()`.

```python
class RedisSession:
    def __init__(self, redis, key): self.redis, self.key = redis, key
    async def get_messages(self, limit=None):
        rows = await self.redis.lrange(self.key, -(limit or 0), -1)
        return [Message.from_dict(json.loads(r)) for r in rows]
    async def add_messages(self, messages):
        await self.redis.rpush(self.key, *[json.dumps(m.to_dict()) for m in messages])
    async def clear(self):
        await self.redis.delete(self.key)
```

## Long-term memory

Give an agent a memory store. It gets two tools, `remember` and `recall`, and the most relevant memories are added to
its instructions at the start of each run.

```python
from openharness import FileMemory

memory = FileMemory("data/memory.json")
agent = Agent(instructions="You are a personal assistant.", memory=memory)

await run(agent, "I'm vegetarian and I live in Toronto.", session=SQLiteSession("mon"))
await run(agent, "Suggest a restaurant for tonight.", session=SQLiteSession("tue"))  # still knows
```

```ts
const memory = new FileMemory("data/memory.json");
const agent = new Agent({ instructions: "You are a personal assistant.", memory });
```

The built-in stores (`InMemoryMemory`, `FileMemory`) rank memories with BM25 keyword scoring and need no extra
services. For semantic search, implement the `Memory` interface (`add`, `search`, `list`, `delete`, `clear`) over your
vector database.

You can also manage memory directly: `await memory.add("Prefers metric units")`, `await memory.search("units")`.
