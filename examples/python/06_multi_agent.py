"""Two multi-agent patterns: handoffs (triage) and agents as tools (manager + workers)."""

import asyncio

from openharness import Agent, run

# --- handoffs: the triage agent passes the whole conversation to a specialist
billing = Agent(name="Billing", description="Refunds, invoices and payments.",
                instructions="You handle billing questions. Be precise about amounts.")
tech = Agent(name="Tech support", description="Bugs, errors and setup help.",
             instructions="You troubleshoot technical problems step by step.")
triage = Agent(name="Triage", instructions="Route the user to the right specialist. Do not answer yourself.",
               handoffs=[billing, tech])

# --- agents as tools: a manager calls workers and combines their answers
researcher = Agent(name="Researcher", instructions="Give 3 short, factual bullet points on the topic.")
critic = Agent(name="Critic", instructions="List the 2 biggest risks or weaknesses of the idea.")
manager = Agent(
    name="Manager",
    instructions="Use the researcher and the critic, then write a 5 sentence recommendation.",
    tools=[researcher.as_tool(), critic.as_tool()],
)


async def main() -> None:
    r = await run(triage, "I was charged twice for my subscription this month.")
    print(f"[{r.last_agent.name}] {r.output}\n")
    r = await run(manager, "Should a 10 person startup self-host its LLM agents?")
    print(r.output)


asyncio.run(main())
