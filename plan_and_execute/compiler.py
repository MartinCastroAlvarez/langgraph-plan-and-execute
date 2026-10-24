"""Variant 3, LLMCompiler-lite: the plan is a DAG, independent steps run in parallel.

``schedule`` finds every step whose dependencies have finished and fans them out with
``Send`` in one superstep (a *wave*); their results come back through the ``results``
reducer, and ``schedule`` runs again. Steps whose dependencies failed are skipped.
When nothing is left, the ``joiner`` model either answers or asks for a new plan,
which goes back to the planner (bounded by ``max_replans``).

"Lite": tasks are dispatched in waves (superstep barriers), not the instant their
inputs are ready, and the planner doesn't stream tasks while it is still writing.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer
from langgraph.types import Send

from plan_and_execute.common import make_finalizer
from plan_and_execute.common import make_planner
from plan_and_execute.common import make_review
from plan_and_execute.common import model_text
from plan_and_execute.config import Settings
from plan_and_execute.plan import PlanError
from plan_and_execute.plan import Step
from plan_and_execute.plan import parse_decision
from plan_and_execute.prompts import JOINER
from plan_and_execute.prompts import data_block
from plan_and_execute.rewoo import run_step
from plan_and_execute.state import PlanState
from plan_and_execute.state import StepResult
from plan_and_execute.state import TaskInput
from plan_and_execute.state import emit
from plan_and_execute.state import plan_of
from plan_and_execute.state import results_by_id
from plan_and_execute.state import short as _short
from plan_and_execute.tools import Toolbox

__all__ = ["build_compiler"]


def build_compiler(
    model: BaseChatModel,
    toolbox: Toolbox,
    settings: Settings,
    *,
    json_model: BaseChatModel | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[PlanState, None, PlanState, PlanState]:
    budget = settings.budget
    planning_model = json_model or model

    def schedule(state: PlanState) -> dict[str, Any]:
        round_ = state.get("round", 1)
        plan = plan_of(state)
        done = results_by_id(state, round_)
        skipped: list[StepResult] = []
        changed = True
        while changed:  # skips cascade: a step after a skipped step is skipped too
            changed = False
            for step in plan.steps:
                if step.id in done:
                    continue
                failed = [d for d in step.depends_on if d in done and not done[d]["ok"]]
                if failed:
                    result: StepResult = {
                        "id": step.id,
                        "round": round_,
                        "tool": step.tool,
                        "input": step.input or "",
                        "output": f"skipped: depends on failed step(s) {failed}",
                        "ok": False,
                        "tool_calls": 0,
                    }
                    done[step.id] = result
                    skipped.append(result)
                    changed = True
        ready = [
            s.id
            for s in plan.steps
            if s.id not in done and all(d in done and done[d]["ok"] for d in s.depends_on)
        ]
        update: dict[str, Any] = {"results": skipped}
        allowed = budget.max_tool_calls - state.get("tool_calls", 0)
        if len(ready) > allowed:
            ready = ready[: max(allowed, 0)]
            update |= {
                "status": "budget",
                "stop_reason": f"max_tool_calls={budget.max_tool_calls} reached",
            }
        if ready:
            emit({"event": "wave", "round": round_, "steps": ready})
            update["waves"] = [ready]
        update["ready"] = ready
        return update

    def dispatch(state: PlanState) -> list[Send] | str:
        ready = state.get("ready", [])
        if not ready:
            return "finalize" if state.get("status") == "budget" else "joiner"
        round_ = state.get("round", 1)
        steps = plan_of(state).by_id()
        done = results_by_id(state, round_)
        sends = []
        for i in ready:
            step = steps[i]
            payload: TaskInput = {
                "step": step.model_dump(),
                "evidence": {d: done[d]["output"] for d in step.depends_on},
                "round": round_,
            }
            sends.append(Send("task", payload))
        return sends

    async def task(state: TaskInput) -> dict[str, Any]:
        step = Step.model_validate(state["step"])
        evidence: dict[int, StepResult] = {
            int(i): {
                "id": int(i),
                "round": state["round"],
                "tool": None,
                "input": "",
                "output": out,
                "ok": True,
                "tool_calls": 0,
            }
            for i, out in state["evidence"].items()
        }
        result = await run_step(toolbox, step, evidence, state["round"])
        emit(
            {
                "event": "step_done",
                "id": step.id,
                "ok": result["ok"],
                "output": _short(result["output"]),
            }
        )
        return {"results": [result], "steps_run": 1, "tool_calls": result["tool_calls"]}

    async def joiner(state: PlanState) -> dict[str, Any]:
        round_ = state.get("round", 1)
        replans = state.get("replans", 0)
        done = results_by_id(state, round_)
        view = [
            {
                "id": s.id,
                "goal": s.goal,
                "input": done[s.id]["input"],
                "result": done[s.id]["output"],
                "ok": done[s.id]["ok"],
            }
            for s in plan_of(state).steps
            if s.id in done
        ]
        prompt = "\n".join(
            [
                data_block("task", state["task"]),
                data_block("results", view),
                data_block("replans_left", str(budget.max_replans - replans)),
            ]
        )
        reply = await planning_model.ainvoke([SystemMessage(JOINER), HumanMessage(prompt)])
        try:
            decision = parse_decision(model_text(reply), allowed={"finish", "replan"})
        except PlanError as exc:
            emit({"event": "decision_invalid", "error": str(exc)})
            return {"history": [{"event": "decision_invalid", "error": str(exc)}]}
        if decision.action == "finish":
            emit({"event": "finish"})
            return {"answer": decision.answer or "", "status": "done"}
        if replans >= budget.max_replans:
            emit({"event": "replan_refused"})
            return {
                "status": "budget",
                "stop_reason": f"max_replans={budget.max_replans} reached",
                "history": [{"event": "refused", "action": "replan"}],
            }
        emit({"event": "replan", "reason": decision.reason})
        return {
            "round": round_ + 1,
            "replans": replans + 1,
            "replan_reason": decision.reason,
            "status": "replanning",
            "history": [{"event": "replan", "round": round_, "reason": decision.reason}],
        }

    def after_planning(state: PlanState) -> str:
        if state.get("status") != "running":
            return END
        return "review" if state.get("round", 1) == 1 else "schedule"

    def after_review(state: PlanState) -> str:
        return "schedule" if state.get("status") == "running" else END

    def after_joiner(state: PlanState) -> str:
        status = state.get("status")
        if status == "done":
            return END
        return "planner" if status == "replanning" else "finalize"

    graph = StateGraph(PlanState)
    graph.add_node("planner", make_planner(planning_model, settings, require_tools=True))
    graph.add_node("review", make_review(settings, require_tools=True))
    graph.add_node("schedule", schedule)
    graph.add_node("task", task, input_schema=TaskInput)
    graph.add_node("joiner", joiner)
    graph.add_node("finalize", make_finalizer())
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", after_planning, ["review", "schedule", END])
    graph.add_conditional_edges("review", after_review, ["schedule", END])
    graph.add_conditional_edges("schedule", dispatch, ["task", "joiner", "finalize"])
    graph.add_edge("task", "schedule")
    graph.add_conditional_edges("joiner", after_joiner, ["planner", "finalize", END])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, name="compiler")
