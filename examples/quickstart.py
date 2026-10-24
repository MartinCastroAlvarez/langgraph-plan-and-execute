"""Run all three variants offline, then approve-and-edit a plan with interrupt().

    python -m examples.quickstart

Uses the fake model, so it needs no ollama and no key. Swap ``provider="fake"`` for
``"ollama"`` (and ``ollama pull qwen3:4b``) to run it on a local model.
"""

from __future__ import annotations

import asyncio
from datetime import date

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from plan_and_execute import Settings
from plan_and_execute import build_graph
from plan_and_execute.progress import CallCounter

TASK = (
    "What weekday is the Project Atlas launch, and how many miles is the Berlin to Lisbon flight?"
)
FREEZER = "How cold is the sample freezer in Fahrenheit?"


async def compare() -> None:
    settings = Settings(provider="fake", today=date(2026, 10, 24))
    for variant in ("classic", "rewoo", "compiler"):
        counter = CallCounter()
        final = await build_graph(variant, settings).ainvoke(
            {"task": TASK}, {"callbacks": [counter], "recursion_limit": 100}
        )
        print(f"{variant:>8}: {counter.total:2d} model calls, {final['tool_calls']} tool calls")
        print(f"          {final['answer']}")
        assert "2026-12-18 (Friday)" in final["answer"]


async def review() -> None:
    settings = Settings(provider="fake", approve_plan=True)
    graph = build_graph("compiler", settings, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "demo"}, "recursion_limit": 100}

    paused = await graph.ainvoke({"task": FREEZER}, config)
    plan = paused["__interrupt__"][0].value["steps"]
    print("\nplan awaiting review:", [f"{s['tool']}({s['input']})" for s in plan])

    plan[1]["input"] = "#E1 to F"  # the reviewer fixes the bad unit before anything runs
    final = await graph.ainvoke(Command(resume={"action": "edit", "steps": plan}), config)
    print("after the edit:", final["answer"])
    assert final["answer"].endswith("-112 °F.")


if __name__ == "__main__":
    asyncio.run(compare())
    asyncio.run(review())
