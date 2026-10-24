"""Human-in-the-loop: interrupt() before execution, resume with Command(resume=...)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import is_invalid_resume
from langgraph.types import Command
from pydantic import ValidationError

from plan_and_execute.config import Budget
from plan_and_execute.fake import FakePlanModel
from plan_and_execute.tools import Toolbox
from tests.conftest import FREEZER
from tests.conftest import graph_for
from tests.conftest import make_settings
from tests.conftest import plan_json
from tests.conftest import run
from tests.conftest import step

CONFIG: dict[str, Any] = {"configurable": {"thread_id": "review"}, "recursion_limit": 100}
FIXED = [step(1, "search_notes", "sample freezer setpoint"), step(2, "convert_units", "#E1 to F")]


def _paused(graph: Any) -> dict[str, Any]:
    first = run(graph, {"task": FREEZER}, CONFIG)
    (interrupt,) = first["__interrupt__"]
    payload: dict[str, Any] = interrupt.value
    return payload


@pytest.mark.parametrize("variant", ["classic", "rewoo", "compiler"])
def test_execution_waits_for_approval(variant: str, toolbox: Toolbox) -> None:
    model = FakePlanModel()
    graph = graph_for(variant, make_settings(approve_plan=True), toolbox, model, InMemorySaver())
    payload = _paused(graph)
    assert payload["kind"] == "plan_review" and len(payload["steps"]) == 2
    assert toolbox.calls == [] and model.calls == ["planner"]
    assert asyncio.run(graph.aget_state(CONFIG)).next == ("review",)
    final = run(graph, Command(resume={"action": "approve"}), CONFIG)
    assert final["status"] == "done" and toolbox.calls
    assert {"event": "approved"} in final["history"]


def test_an_edited_plan_replaces_the_planners(toolbox: Toolbox) -> None:
    graph = graph_for("rewoo", make_settings(approve_plan=True), toolbox, None, InMemorySaver())
    _paused(graph)
    final = run(graph, Command(resume={"action": "edit", "steps": FIXED}), CONFIG)
    assert final["results"][1]["output"] == "-112 °F"
    assert any(h["event"] == "edited" for h in final["history"])


def test_reject_ends_without_running_anything(toolbox: Toolbox) -> None:
    graph = graph_for("classic", make_settings(approve_plan=True), toolbox, None, InMemorySaver())
    _paused(graph)
    final = run(graph, Command(resume={"action": "reject", "reason": "too risky"}), CONFIG)
    assert final["status"] == "rejected" and final["answer"] == "The plan was rejected. too risky"
    assert toolbox.calls == []


def test_a_malformed_answer_raises_and_the_interrupt_stays_open(toolbox: Toolbox) -> None:
    graph = graph_for("classic", make_settings(approve_plan=True), toolbox, None, InMemorySaver())
    _paused(graph)
    bad_answers: list[Any] = [
        {"action": "ship it"},
        {"action": "edit"},
        {"action": "edit", "steps": [step(1, None, None, [2])]},
        {"action": "approve", "extra": "field"},
    ]
    for bad in bad_answers:
        with pytest.raises(ValidationError) as info:
            run(graph, Command(resume=bad), CONFIG)
        assert is_invalid_resume(info.value)
        assert asyncio.run(graph.aget_state(CONFIG)).next == ("review",)
    final = run(graph, Command(resume={"action": "approve"}), CONFIG)
    assert final["status"] == "done"


def test_an_edit_over_budget_is_sent_back_with_the_error(toolbox: Toolbox) -> None:
    settings = make_settings(approve_plan=True, budget=Budget(max_steps=2))
    graph = graph_for("compiler", settings, toolbox, None, InMemorySaver())
    _paused(graph)
    too_long = [step(i, "calculator", "1") for i in (1, 2, 3)]
    second = run(graph, Command(resume={"action": "edit", "steps": too_long}), CONFIG)
    (interrupt,) = second["__interrupt__"]
    assert "budget allows 2" in interrupt.value["error"]
    no_tool = [step(1, None, None)]
    third = run(graph, Command(resume={"action": "edit", "steps": no_tool}), CONFIG)
    assert "need a tool" in third["__interrupt__"][0].value["error"]
    final = run(graph, Command(resume={"action": "edit", "steps": FIXED}), CONFIG)
    assert final["answer"].endswith("which is -112 °F.")


def test_three_bad_edits_reject_the_plan(toolbox: Toolbox) -> None:
    settings = make_settings(approve_plan=True, budget=Budget(max_steps=1))
    model = FakePlanModel(script={"planner": [plan_json(step(1, "calculator", "1"))]})
    graph = graph_for("rewoo", settings, toolbox, model, InMemorySaver())
    _paused(graph)
    final: dict[str, Any] = {}
    for _ in range(3):
        final = run(graph, Command(resume={"action": "edit", "steps": FIXED}), CONFIG)
    assert final["status"] == "rejected" and "budget allows 1" in final["answer"]


def test_no_interrupt_when_approval_is_off(toolbox: Toolbox) -> None:
    graph = graph_for("classic", make_settings(), toolbox, None, InMemorySaver())
    final = run(graph, {"task": FREEZER}, CONFIG)
    assert "__interrupt__" not in final and final["status"] == "done"


def test_resume_from_sqlite_in_a_new_graph_instance(tmp_path: Path, toolbox: Toolbox) -> None:
    """The pause survives the process: a second graph + saver on the same file resumes it."""
    db = str(tmp_path / "runs.sqlite")
    settings = make_settings(approve_plan=True)

    async def pause() -> dict[str, Any]:
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = graph_for("compiler", settings, toolbox, None, saver)
            result: dict[str, Any] = await graph.ainvoke({"task": FREEZER}, CONFIG)
            return result

    async def resume() -> dict[str, Any]:
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = graph_for("compiler", settings, toolbox, None, saver)
            result: dict[str, Any] = await graph.ainvoke(
                Command(resume={"action": "edit", "steps": FIXED}), CONFIG
            )
            return result

    assert "__interrupt__" in asyncio.run(pause())
    final = asyncio.run(resume())
    assert final["status"] == "done" and final["waves"] == [[1], [2]]
    assert final["answer"].endswith("which is -112 °F.")
