"""Plan parsing and validation, decisions, and #E placeholder substitution."""

from __future__ import annotations

import json

import pytest

from plan_and_execute.plan import Plan
from plan_and_execute.plan import PlanError
from plan_and_execute.plan import Step
from plan_and_execute.plan import evidence_value
from plan_and_execute.plan import extract_json
from plan_and_execute.plan import parse_decision
from plan_and_execute.plan import parse_plan
from plan_and_execute.plan import substitute
from tests.conftest import plan_json
from tests.conftest import step


def test_parses_fenced_json_with_prose_and_think_block() -> None:
    text = (
        "<think>{not json} let me plan</think>Sure! Here is the plan:\n```json\n"
        + plan_json(step(1, "search_notes", "atlas"), step(2, "date_math", "#E1 + 3 days"))
        + "\n```\nGood luck."
    )
    plan = parse_plan(text, max_steps=5)
    assert [s.id for s in plan.steps] == [1, 2]
    assert plan.steps[1].depends_on == [1]  # derived from the #E1 placeholder


def test_bare_list_is_accepted_and_extra_keys_ignored() -> None:
    raw = json.dumps([{"id": 1, "goal": "look up", "tool": "None", "why": "extra"}])
    plan = parse_plan(raw, max_steps=3)
    assert plan.steps[0].tool is None


def test_tool_names_are_normalised_and_unknown_tools_rejected() -> None:
    assert Step(id=1, goal="g", tool=" Calculator ").tool == "calculator"
    with pytest.raises(PlanError, match="tool"):
        parse_plan(plan_json(step(1, "shell", "rm -rf /")), max_steps=3)


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([step(1, None, None), step(1, None, None)], "unique"),
        ([step(1, None, None, [7])], "unknown steps"),
        ([step(1, None, None, [2]), step(2, None, None, [1])], "cycle"),
        ([step(1, "calculator", "#E1 + 1")], "itself"),
        ([], "at least 1"),
    ],
)
def test_invalid_plans_are_rejected(steps: list[dict[str, object]], message: str) -> None:
    with pytest.raises(PlanError, match=message):
        parse_plan(plan_json(*steps), max_steps=5)


def test_no_json_at_all() -> None:
    with pytest.raises(PlanError, match="no JSON"):
        parse_plan("I would first look things up.", max_steps=3)
    with pytest.raises(PlanError, match="no JSON"):
        extract_json("{broken: json")


def test_budget_and_variant_checks() -> None:
    three = plan_json(*(step(i, "search_notes", "x") for i in (1, 2, 3)))
    with pytest.raises(PlanError, match="budget allows 2"):
        parse_plan(three, max_steps=2)
    with pytest.raises(PlanError, match=r"steps \[1\] need a tool"):
        parse_plan(plan_json(step(1, None, None)), max_steps=3, require_tools=True)


def test_waves_and_order_keep_the_planner_order() -> None:
    plan = Plan.model_validate(
        {
            "steps": [
                step(1, "search_notes", "a"),
                step(2, "search_notes", "b"),
                step(3, "date_math", "#E1 + #E2"),
                step(4, "search_notes", "c"),
                step(5, "convert_units", "#E4 to mi"),
            ]
        }
    )
    assert plan.waves() == [[1, 2, 4], [3, 5]]
    assert plan.order() == [1, 2, 4, 3, 5]
    assert plan.waves(done=[1, 2]) == [[3, 4], [5]]


def test_out_of_order_dependencies_are_topologically_sorted() -> None:
    plan = Plan.model_validate(
        {"steps": [step(2, "calculator", "#E1 * 2"), step(1, "calculator", "1")]}
    )
    assert plan.order() == [1, 2]


def test_decisions() -> None:
    finish = parse_decision('{"action": "FINISH", "answer": "42"}', allowed={"finish"})
    assert finish.answer == "42"
    insert = parse_decision(
        json.dumps({"action": "insert", "steps": [step(9, "calculator", "1+1")]}),
        allowed={"insert"},
    )
    assert insert.steps[0].id == 9
    with pytest.raises(PlanError, match="needs an answer"):
        parse_decision('{"action": "finish"}', allowed={"finish"})
    with pytest.raises(PlanError, match="needs steps"):
        parse_decision('{"action": "replace"}', allowed={"replace"})
    with pytest.raises(PlanError, match="not one of"):
        parse_decision('{"action": "replan"}', allowed={"finish"})
    with pytest.raises(PlanError, match="JSON object"):
        parse_decision("[1, 2]", allowed={"finish"})


def test_evidence_value_takes_the_first_line_without_the_note_id() -> None:
    out = "[offices] Berlin to Lisbon flight distance: 2310 km\n[offices] other fact"
    assert evidence_value(out) == "Berlin to Lisbon flight distance: 2310 km"
    assert evidence_value(out, numeric=True) == "2310"
    assert evidence_value("[x] Budget: 120,000 EUR", numeric=True) == "120000"
    assert evidence_value("x" * 1000) == "x" * 300
    with pytest.raises(PlanError, match="no number"):
        evidence_value("no digits here", numeric=True)


def test_substitute_replaces_every_placeholder() -> None:
    results = {1: "[travel-policy] Per diem: 85 EUR per day", 2: "[project-atlas] Team size: 6"}
    assert substitute("#E1 * #E2 * 4", results, numeric=True) == "85 * 6 * 4"
    assert substitute("#E1 to F", {1: "[lab] setpoint: -80 °C"}) == "setpoint: -80 °C to F"
    assert substitute("#E10 and #E1", {1: "a", 10: "b"}) == "b and a"
    with pytest.raises(PlanError, match="#E3 has no result"):
        substitute("#E3", results)
