"""Sessions remember the conversation. Memory remembers facts across conversations."""

import asyncio

from openharness import Agent, FileMemory, SQLiteSession, run

memory = FileMemory("data/memory.json")  # durable facts, shared by every session
agent = Agent(instructions="You are a personal assistant. Save lasting preferences with remember.", memory=memory)


async def main() -> None:
    monday = SQLiteSession("monday", "data/chats.db")
    await run(agent, "Hi, I'm Sam. I always want temperatures in Celsius.", session=monday)
    print((await run(agent, "What's my name?", session=monday)).output)

    tuesday = SQLiteSession("tuesday", "data/chats.db")  # new conversation, same memory
    print((await run(agent, "Which temperature unit do I prefer?", session=tuesday)).output)
    print("memories:", [m.text for m in await memory.list()])


asyncio.run(main())
