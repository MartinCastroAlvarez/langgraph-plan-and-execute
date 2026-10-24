"""ReWOO: two model calls, literal #E substitution, failures skip their dependants."""

from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from plan_and_execute.config import Budget
from plan_and_execute.fake import FakePlanModel
from plan_and_execute.graphs import build_graph
from plan_and_execute.tools import Toolbox
from tests.conftest import FREEZER
from tests.conftest import LAUNCH
from tests.conftest import PER_DIEM
from tests.conftest import graph_for
from tests.conftest import make_settings
from tests.conftest import plan_json
from tests.conftest import run
from tests.conftest import step


def test_two_model_calls_and_substituted_tool_inputs(toolbox: Toolbox) -> None:
    model = FakePlanModel()
    final = run(graph_for("rewoo", make_settings(), toolbox, model), {"task": LAUNCH})
    assert model.calls == ["planner", "solver"]
    inputs = dict(toolbox.calls[i] for i in (3, 4))
    assert inputs["date_math"] == (
        "Project Atlas kickoff date: 2026-11-03 + Project Atlas launch: 45 days after kickoff"
    )
    assert inputs["convert_units"] == "Berlin to Lisbon flight distance: 2310 km to mi"
    assert final["answer"] == (
        "Project Atlas launches on 2026-12-18 (Friday), and the Berlin to Lisbon flight is "
        "1435.37 mi."
    )


def test_calculator_placeholders_become_numbers(toolbox: Toolbox) -> None:
    final = run(graph_for("rewoo", make_settings(), toolbox), {"task": PER_DIEM})
    calc = next(r for r in final["results"] if r["tool"] == "calculator")
    assert calc["input"] == "85 * 6 * 4" and calc["output"] == "2040"


def test_a_failed_step_skips_its_dependants_and_there_is_no_replan(toolbox: Toolbox) -> None:
    model = FakePlanModel(
        script={
            "planner": [
                plan_json(
                    step(1, "convert_units", "5 km to degrees"),
                    step(2, "calculator", "#E1 * 2"),
                    step(3, "calculator", "#E2 + 1"),
                    step(4, "calculator", "2 + 2"),
                )
            ]
        }
    )
    final = run(graph_for("rewoo", make_settings(), toolbox, model), {"task": "t"})
    out = {r["id"]: r["output"] for r in final["results"]}
    assert out[1].startswith("error: unknown target unit")
    assert out[2] == "skipped: depends on failed step(s) [1]"
    assert out[3] == "skipped: depends on failed step(s) [2]"
    assert out[4] == "4"
    assert model.calls == ["planner", "solver"] and final["tool_calls"] == 2


def test_the_recovery_scenario_cannot_recover(toolbox: Toolbox) -> None:
    final = run(graph_for("rewoo", make_settings(), toolbox), {"task": FREEZER})
    assert final["answer"].startswith("I could only find part of the answer")


def test_steps_without_a_tool_are_rejected_and_the_planner_repairs_them(
    toolbox: Toolbox,
) -> None:
    model = FakePlanModel(
        script={"planner": [plan_json(step(1, None, None)), plan_json(step(1, "calculator", "1"))]}
    )
    final = run(graph_for("rewoo", make_settings(), toolbox, model), {"task": "t"})
    assert model.calls == ["planner", "planner", "solver"]
    assert final["results"][0]["output"] == "1"


def test_tool_budget_stops_the_worker_and_the_solver_still_answers(toolbox: Toolbox) -> None:
    settings = make_settings(budget=Budget(max_tool_calls=1))
    model = FakePlanModel(
        script={"planner": [plan_json(step(1, "calculator", "1"), step(2, "calculator", "2"))]}
    )
    final = run(graph_for("rewoo", settings, toolbox, model), {"task": "t"})
    assert final["status"] == "budget" and final["tool_calls"] == 1
    assert final["answer"]


def test_with_langchains_generic_fake_chat_model(toolbox: Toolbox) -> None:
    """The graph only needs a BaseChatModel: LangChain's own fake drives it too."""
    plan = plan_json(step(1, "search_notes", "next all-hands"), step(2, "date_math", "#E1"))
    model = GenericFakeChatModel(
        messages=iter([AIMessage(f"```json\n{plan}\n```"), AIMessage("<think>hm</think>Friday.")])
    )
    graph = build_graph("rewoo", make_settings(), model=model, toolbox=toolbox)
    final = run(graph, {"task": "When is the next all-hands meeting?"})
    assert final["results"][1]["output"] == "2026-11-20 (Friday)"
    assert final["answer"] == "Friday."
    assert json.loads(json.dumps(final["steps"]))  # state stays JSON-native
