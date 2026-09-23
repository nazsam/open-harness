"""Test agent logic without any API key using FakeModel."""

import asyncio

from openharness import Agent, FakeModel, call, run, tool


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order."""
    return {"id": order_id, "status": "shipped"}


async def main() -> None:
    model = FakeModel([call("lookup_order", order_id="A123"), "Your order A123 has shipped."])
    result = await run(Agent(model=model, tools=[lookup_order]), "Where is order A123?")
    assert result.output == "Your order A123 has shipped."
    assert result.new_messages[2].content == '{"id": "A123", "status": "shipped"}'
    print("ok:", result.output)


asyncio.run(main())
