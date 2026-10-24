"""Shared fixtures: fake-provider settings, a toolbox over the bundled notes, run helpers."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

import pytest

from plan_and_execute.config import Budget
from plan_and_execute.config import Settings
from plan_and_execute.fake import FakePlanModel
from plan_and_execute.graphs import build_graph
from plan_and_execute.notes import Notes
from plan_and_execute.tools import Toolbox

LAUNCH = (
    "What weekday is the Project Atlas launch, and how many miles is the Berlin to Lisbon flight?"
)
PER_DIEM = "What do 4 days of Lisbon per diem cost for the Atlas team?"
FREEZER = "How cold is the sample freezer in Fahrenheit?"
TODAY = date(2026, 10, 24)


@pytest.fixture(scope="session")
def notes() -> Notes:
    return Notes.from_dir()


@pytest.fixture
def toolbox(notes: Notes) -> Toolbox:
    return Toolbox(notes=notes, today=TODAY)


def make_settings(**changes: Any) -> Settings:
    budget = changes.pop("budget", Budget())
    return Settings(provider="fake", today=TODAY, budget=budget, **changes)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


def step(
    id: int, tool: str | None, input: str | None, deps: list[int] | None = None
) -> dict[str, Any]:
    return {
        "id": id,
        "goal": f"step {id}",
        "tool": tool,
        "input": input,
        "success": "",
        "depends_on": deps or [],
    }


def plan_json(*steps: dict[str, Any]) -> str:
    return json.dumps({"steps": list(steps)})


def graph_for(
    variant: str,
    settings: Settings,
    toolbox: Toolbox,
    model: Any | None = None,
    checkpointer: Any = None,
) -> Any:
    return build_graph(
        variant,
        settings,
        model=model or FakePlanModel(),
        toolbox=toolbox,
        checkpointer=checkpointer,
    )


def run(graph: Any, inputs: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = asyncio.run(graph.ainvoke(inputs, config or {"recursion_limit": 100}))
    return result
