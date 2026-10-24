"""Classic plan-and-execute: execution order, replan triggers, budgets, failure handling."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from plan_and_execute.classic import build_executor
from plan_and_execute.config import Budget
from plan_and_execute.fake import FakePlanModel
from plan_and_execute.tools import Toolbox
from tests.conftest import FREEZER
from tests.conftest import LAUNCH
from tests.conftest import graph_for
from tests.conftest import make_settings
from tests.conftest import plan_json
from tests.conftest import run
from tests.conftest import step


def test_happy_path_runs_every_step_and_replans_after_each(toolbox: Toolbox) -> None:
    model = FakePlanModel()
    final = run(graph_for("classic", make_settings(), toolbox, model), {"task": LAUNCH})
    assert final["status"] == "done"
    assert "2026-12-18 (Friday)" in final["answer"] and "1435.37 mi" in final["answer"]
    assert [r["id"] for r in final["results"]] == [1, 2, 4, 3, 5]  # topological order
    assert all(r["ok"] for r in final["results"])
    # 1 plan + (tool call + answer) per step + 1 replan per step
    assert model.calls.count("planner") == 1
    assert model.calls.count("executor") == 10
    assert model.calls.count("replanner") == 5
    assert final["tool_calls"] == 5 and final["steps_run"] == 5


def test_failed_step_triggers_an_inserted_recovery_step(toolbox: Toolbox) -> None:
    final = run(graph_for("classic", make_settings(), toolbox), {"task": FREEZER})
    by_id = {r["id"]: r for r in final["results"]}
    assert by_id[2]["ok"] is False and "unknown target unit" in by_id[2]["output"]
    assert by_id[3]["ok"] is True and by_id[3]["output"] == "-112 °F"
    assert [s["id"] for s in final["steps"]] == [1, 2, 3]
    insert = [h for h in final["history"] if h["event"] == "insert"]
    assert insert == [
        {"event": "insert", "after": 2, "steps": [3], "reason": "the last step failed"}
    ]
    assert final["replans"] == 1
    assert final["answer"].endswith("which is -112 °F.")


def test_replace_drops_the_remaining_steps(toolbox: Toolbox) -> None:
    model = FakePlanModel(
        script={
            "planner": [
                plan_json(
                    step(1, "calculator", "1 + 1"),
                    step(2, "calculator", "2 + 2"),
                    step(3, "calculator", "3 + 3"),
                )
            ],
            "replanner": [
                json.dumps(
                    {"action": "replace", "steps": [step(4, "calculator", "#E1 * 10", [1])]}
                ),
            ],
        }
    )
    final = run(graph_for("classic", make_settings(), toolbox, model), {"task": "sums"})
    assert [r["id"] for r in final["results"]] == [1, 4]
    assert final["results"][-1]["output"] == "20"
    assert [s["id"] for s in final["steps"]] == [1, 4]


def test_max_steps_budget_ends_gracefully_with_a_partial_answer(toolbox: Toolbox) -> None:
    settings = make_settings(budget=Budget(max_steps=2))
    model = FakePlanModel(
        script={"planner": [plan_json(*(step(i, "calculator", f"{i} + {i}") for i in (1, 2)))]}
    )
    # the replanner keeps inserting steps; the step budget stops the run
    model.script["replanner"] = [
        json.dumps({"action": "insert", "steps": [step(10 + i, "calculator", "1")]})
        for i in range(5)
    ]
    final = run(graph_for("classic", settings, toolbox, model), {"task": "loop"})
    assert final["status"] == "budget"
    assert final["steps_run"] == 2
    assert final["answer"].startswith("Stopped early (max_steps=2 reached). Found:")


def test_max_tool_calls_budget(toolbox: Toolbox) -> None:
    settings = make_settings(budget=Budget(max_tool_calls=2))
    model = FakePlanModel(
        script={"planner": [plan_json(*(step(i, "calculator", "1 + 1") for i in (1, 2, 3)))]}
    )
    final = run(graph_for("classic", settings, toolbox, model), {"task": "three sums"})
    assert final["status"] == "budget"
    assert final["tool_calls"] == 2 and "max_tool_calls=2" in final["stop_reason"]


def test_max_replans_refuses_further_plan_edits(toolbox: Toolbox) -> None:
    settings = make_settings(budget=Budget(max_replans=0))
    final = run(graph_for("classic", settings, toolbox), {"task": FREEZER})
    assert final["replans"] == 0
    assert {"event": "refused", "action": "insert"} in final["history"]
    assert [r["id"] for r in final["results"]] == [1, 2]
    assert final["status"] == "done"  # finalize composes an answer from what was found
    assert "Could not complete" in final["answer"]


def test_invalid_replanner_output_is_treated_as_continue(toolbox: Toolbox) -> None:
    model = FakePlanModel(
        script={
            "planner": [
                plan_json(step(1, "calculator", "6 * 7"), step(2, "calculator", "#E1 + 1"))
            ],
            "replanner": [
                "no idea",
                json.dumps({"action": "insert", "steps": [step(1, "calculator", "dup id")]}),
            ],
        }
    )
    final = run(graph_for("classic", make_settings(), toolbox, model), {"task": "t"})
    assert [h["event"] for h in final["history"]].count("decision_invalid") == 2
    assert [r["output"] for r in final["results"]] == ["42", "43"]
    assert final["answer"] == "Found: step 1: 42; step 2: 43."


def test_planner_gets_one_repair_attempt(toolbox: Toolbox) -> None:
    model = FakePlanModel(
        script={"planner": ["not json", plan_json(step(1, "search_notes", "freezer"))]}
    )
    final = run(graph_for("classic", make_settings(), toolbox, model), {"task": "freezer?"})
    assert model.calls.count("planner") == 2 and final["status"] == "done"


def test_planner_that_never_produces_a_plan_fails_cleanly(toolbox: Toolbox) -> None:
    model = FakePlanModel(script={"planner": ["nope", plan_json(step(1, "rm", "-rf"))]})
    final = run(graph_for("classic", make_settings(), toolbox, model), {"task": "t"})
    assert final["status"] == "failed"
    assert "I could not make a valid plan" in final["answer"]
    assert "executor" not in model.calls


def test_a_model_error_fails_the_step_not_the_run(toolbox: Toolbox) -> None:
    model = FakePlanModel(
        script={
            "planner": [plan_json(step(1, "calculator", "1 + 1"), step(2, "calculator", "2 + 2"))],
            "executor": [ConnectionError("model went away")],
        }
    )
    final = run(graph_for("classic", make_settings(), toolbox, model), {"task": "t"})
    first = final["results"][0]
    assert first["ok"] is False and first["output"].startswith("FAILED: ConnectionError")
    assert final["results"][1]["ok"] is True


def test_a_tool_free_step_and_a_step_the_agent_reports_failed(toolbox: Toolbox) -> None:
    model = FakePlanModel(
        script={
            "planner": [plan_json(step(1, None, None), step(2, None, None))],
            "executor": ["", AIMessage("FAILED: cannot")],
        }
    )
    final = run(graph_for("classic", make_settings(), toolbox, model), {"task": "t"})
    assert [r["ok"] for r in final["results"]] == [False, False]
    assert final["results"][0]["output"] == "FAILED: no answer"


def test_per_step_tool_cap_comes_from_create_agent_middleware(toolbox: Toolbox) -> None:
    calls = [
        AIMessage("", tool_calls=[{"name": "calculator", "args": {"input": "1"}, "id": f"c{i}"}])
        for i in range(5)
    ]
    model = FakePlanModel(script={"executor": calls})
    agent = build_executor(model, toolbox, tool_limit=2)
    out = run(agent, {"messages": [("user", "go")]})
    tool_messages = [m for m in out["messages"] if m.type == "tool" and m.status != "error"]
    assert len(tool_messages) == 2
    assert {"model", "tools"} <= set(agent.get_graph().nodes)
