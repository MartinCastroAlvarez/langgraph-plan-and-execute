"""The graph state shared by the three variants, and small helpers to read it.

The state holds only JSON-native values (plans are ``Step.model_dump()`` dicts), so any
checkpointer can store it without custom serialisers. Keys written by parallel tasks
(``results``, ``tool_calls``, ``waves``) have reducers.
"""

from __future__ import annotations

import operator
from typing import Annotated
from typing import Any
from typing import TypedDict

from langgraph.config import get_stream_writer

from plan_and_execute.plan import Plan

__all__ = ["PlanState", "StepResult", "TaskInput", "emit", "plan_of", "results_by_id", "short"]


class StepResult(TypedDict):
    id: int
    round: int
    tool: str | None
    input: str
    output: str
    ok: bool
    tool_calls: int


class PlanState(TypedDict, total=False):
    task: str
    steps: list[dict[str, Any]]
    """The current plan, ``Step.model_dump()`` per step (done and pending)."""
    pending: list[int]
    """Classic only: ids still to execute, in order."""
    results: Annotated[list[StepResult], operator.add]
    round: int
    """Compiler only: 1 for the first plan, +1 per joiner re-plan."""
    ready: list[int]
    waves: Annotated[list[list[int]], operator.add]
    """Compiler only: the step ids dispatched together, wave by wave."""
    tool_calls: Annotated[int, operator.add]
    steps_run: Annotated[int, operator.add]
    replans: int
    history: Annotated[list[dict[str, Any]], operator.add]
    """Plan events: planned, approved, edited, insert, replace, replan, refused..."""
    status: str
    """running | done | budget | rejected | failed"""
    stop_reason: str
    replan_reason: str
    answer: str


class TaskInput(TypedDict):
    """What the compiler's ``Send`` hands each parallel task."""

    step: dict[str, Any]
    evidence: dict[int, str]
    round: int


def plan_of(state: PlanState) -> Plan:
    return Plan.model_validate({"steps": state.get("steps", [])})


def results_by_id(state: PlanState, round_: int | None = None) -> dict[int, StepResult]:
    """The latest result per step id (of one round, when given)."""
    out: dict[int, StepResult] = {}
    for result in state.get("results", []):
        if round_ is None or result["round"] == round_:
            out[result["id"]] = result
    return out


def emit(event: dict[str, Any]) -> None:
    """Send a ``custom`` stream event; a no-op when the run is not being streamed."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(event)


def short(text: str, limit: int = 120) -> str:
    """The first line of ``text``, cut to ``limit`` characters (for progress events)."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first if len(first) <= limit else first[: limit - 3] + "..."
