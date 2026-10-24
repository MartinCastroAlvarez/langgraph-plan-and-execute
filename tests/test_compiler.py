"""LLMCompiler-lite: DAG waves, real parallelism, skip cascades, joiner re-plans, budgets."""

from __future__ import annotations

import asyncio
import json
import time

from plan_and_execute.config import Budget
from plan_and_execute.fake import FakePlanModel
from plan_and_execute.notes import Notes
from plan_and_execute.tools import Toolbox
from tests.conftest import FREEZER
from tests.conftest import LAUNCH
from tests.conftest import graph_for
from tests.conftest import make_settings
from tests.conftest import plan_json
from tests.conftest import run
from tests.conftest import step


def test_independent_steps_are_dispatched_together(toolbox: Toolbox) -> None:
    model = FakePlanModel()
    final = run(graph_for("compiler", make_settings(), toolbox, model), {"task": LAUNCH})
    assert final["waves"] == [[1, 2, 4], [3, 5]]
    started = [name for name, _ in toolbox.calls]
    assert started[:3] == ["search_notes"] * 3
    assert sorted(started[3:]) == ["convert_units", "date_math"]
    assert model.calls == ["planner", "joiner"]
    assert "2026-12-18 (Friday)" in final["answer"]


def test_waves_really_run_in_parallel(notes: Notes) -> None:
    def elapsed(variant: str) -> tuple[float, int]:
        box = Toolbox(notes=notes, latency=0.2)
        graph = graph_for(variant, make_settings(), box)
        start = time.perf_counter()
        run(graph, {"task": LAUNCH})
        return time.perf_counter() - start, box.peak_in_flight

    parallel, peak = elapsed("compiler")
    sequential, seq_peak = elapsed("rewoo")
    assert peak == 3 and seq_peak == 1
    assert parallel < 0.7 < sequential  # 2 waves x 0.2 s vs 5 steps x 0.2 s


def test_skips_cascade_through_the_dag(toolbox: Toolbox) -> None:
    model = FakePlanModel(
        script={
            "planner": [
                plan_json(
                    step(1, "calculator", "1 / 0"),
                    step(2, "calculator", "#E1 + 1"),
                    step(3, "calculator", "#E2 + 1"),
                    step(4, "calculator", "5"),
                )
            ],
            "joiner": [json.dumps({"action": "finish", "answer": "partial"})],
        }
    )
    final = run(graph_for("compiler", make_settings(), toolbox, model), {"task": "t"})
    out = {r["id"]: r["output"] for r in final["results"]}
    assert out == {
        1: "error: division by zero",
        2: "skipped: depends on failed step(s) [1]",
        3: "skipped: depends on failed step(s) [2]",
        4: "5",
    }
    assert final["waves"] == [[1, 4]]


def test_joiner_replans_after_a_failure(toolbox: Toolbox) -> None:
    model = FakePlanModel()
    final = run(graph_for("compiler", make_settings(), toolbox, model), {"task": FREEZER})
    assert model.calls == ["planner", "joiner", "planner", "joiner"]
    assert final["round"] == 2 and final["replans"] == 1
    assert final["waves"] == [[1], [2], [1], [2]]
    assert final["answer"].endswith("which is -112 °F.")


def test_replan_budget_finalizes_instead_of_looping(toolbox: Toolbox) -> None:
    always_replan = [json.dumps({"action": "replan", "reason": "again"})] * 5
    model = FakePlanModel(
        script={
            "planner": [plan_json(step(1, "calculator", "1 / 0"))] * 5,
            "joiner": always_replan,
        }
    )
    settings = make_settings(budget=Budget(max_replans=2))
    final = run(graph_for("compiler", settings, toolbox, model), {"task": "t"})
    assert model.calls.count("planner") == 3 and model.calls.count("joiner") == 3
    assert final["status"] == "budget" and "max_replans=2" in final["stop_reason"]
    assert final["answer"].startswith("Stopped early (max_replans=2 reached).")


def test_tool_budget_truncates_a_wave(toolbox: Toolbox) -> None:
    model = FakePlanModel(
        script={"planner": [plan_json(*(step(i, "calculator", str(i)) for i in (1, 2, 3)))]}
    )
    settings = make_settings(budget=Budget(max_tool_calls=2))
    final = run(graph_for("compiler", settings, toolbox, model), {"task": "t"})
    assert final["waves"] == [[1, 2]]
    assert final["status"] == "budget" and final["tool_calls"] == 2
    assert "joiner" not in model.calls


def test_invalid_joiner_output_finalizes(toolbox: Toolbox) -> None:
    model = FakePlanModel(
        script={"planner": [plan_json(step(1, "calculator", "2 * 21"))], "joiner": ["?"]}
    )
    final = run(graph_for("compiler", make_settings(), toolbox, model), {"task": "t"})
    assert final["status"] == "done" and final["answer"] == "Found: step 1: 42."


def test_concurrent_runs_do_not_share_state(toolbox: Toolbox) -> None:
    graph = graph_for("compiler", make_settings(), toolbox)

    async def both() -> list[dict[str, object]]:
        return list(
            await asyncio.gather(graph.ainvoke({"task": LAUNCH}), graph.ainvoke({"task": FREEZER}))
        )

    launch, freezer = asyncio.run(both())
    assert "1435.37 mi" in str(launch["answer"]) and "-112 °F" in str(freezer["answer"])
