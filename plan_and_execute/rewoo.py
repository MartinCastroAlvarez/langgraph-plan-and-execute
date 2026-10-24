"""Variant 2, ReWOO (Reasoning WithOut Observation): plan once, run tools, solve once.

The planner writes every step with ``#E<n>`` evidence placeholders. The worker runs the
steps in dependency order and substitutes each placeholder with the evidence it stands
for, with no model call in between. The solver answers from the evidence. Two model
calls per task, whatever the number of steps, but no chance to react to a failure.
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

from plan_and_execute.common import make_planner
from plan_and_execute.common import make_review
from plan_and_execute.common import model_text
from plan_and_execute.config import Settings
from plan_and_execute.plan import PlanError
from plan_and_execute.plan import Step
from plan_and_execute.plan import substitute
from plan_and_execute.prompts import SOLVER
from plan_and_execute.prompts import data_block
from plan_and_execute.state import PlanState
from plan_and_execute.state import StepResult
from plan_and_execute.state import emit
from plan_and_execute.state import plan_of
from plan_and_execute.state import results_by_id
from plan_and_execute.state import short as _short
from plan_and_execute.tools import Toolbox
from plan_and_execute.tools import ToolError

__all__ = ["build_rewoo", "run_step"]

MAX_OUTPUT_CHARS = 1000
MAX_ANSWER_CHARS = 4000


async def run_step(
    toolbox: Toolbox, step: Step, evidence: dict[int, StepResult], round_: int = 1
) -> StepResult:
    """Run one fully specified step: skip it if a dependency failed, else substitute + call."""
    failed = [d for d in step.depends_on if d in evidence and not evidence[d]["ok"]]
    base: StepResult = {
        "id": step.id,
        "round": round_,
        "tool": step.tool,
        "input": step.input or "",
        "output": "",
        "ok": False,
        "tool_calls": 0,
    }
    if failed:
        return base | {"output": f"skipped: depends on failed step(s) {failed}"}
    try:
        text = substitute(
            step.input or "",
            {i: r["output"] for i, r in evidence.items()},
            numeric=step.tool == "calculator",
        )
    except PlanError as exc:
        return base | {"output": f"error: {exc}"}
    try:
        output = await toolbox.run(step.tool or "", text)
    except ToolError as exc:
        return base | {"input": text, "output": f"error: {exc}", "tool_calls": 1}
    return base | {"input": text, "output": output[:MAX_OUTPUT_CHARS], "ok": True, "tool_calls": 1}


def build_rewoo(
    model: BaseChatModel,
    toolbox: Toolbox,
    settings: Settings,
    *,
    json_model: BaseChatModel | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[PlanState, None, PlanState, PlanState]:
    budget = settings.budget

    async def worker(state: PlanState) -> dict[str, Any]:
        plan = plan_of(state)
        done = results_by_id(state)
        todo = [i for i in plan.order() if i not in done]
        if state.get("tool_calls", 0) >= budget.max_tool_calls:
            return {
                "status": "budget",
                "stop_reason": f"max_tool_calls={budget.max_tool_calls} reached",
            }
        step = plan.by_id()[todo[0]]
        result = await run_step(toolbox, step, done)
        emit(
            {
                "event": "step_done",
                "id": step.id,
                "ok": result["ok"],
                "input": result["input"],
                "output": _short(result["output"]),
            }
        )
        return {"results": [result], "steps_run": 1, "tool_calls": result["tool_calls"]}

    async def solver(state: PlanState) -> dict[str, Any]:
        done = results_by_id(state)
        evidence = [
            {
                "id": s.id,
                "goal": s.goal,
                "tool": s.tool,
                "input": done[s.id]["input"] if s.id in done else s.input,
                "evidence": done[s.id]["output"] if s.id in done else "not run",
            }
            for s in plan_of(state).steps
        ]
        prompt = data_block("task", state["task"]) + "\n" + data_block("evidence", evidence)
        if state.get("status") == "budget":
            prompt += "\n" + data_block("stopped", state.get("stop_reason", ""))
        reply = await model.ainvoke([SystemMessage(SOLVER), HumanMessage(prompt)])
        answer = model_text(reply)[:MAX_ANSWER_CHARS]
        status = state.get("status", "running")
        return {"answer": answer, "status": "done" if status == "running" else status}

    def after_planning(state: PlanState) -> str:
        return "review" if state.get("status") == "running" else END

    def after_review(state: PlanState) -> str:
        return "worker" if state.get("status") == "running" else END

    def after_worker(state: PlanState) -> str:
        if state.get("status") == "budget":
            return "solver"
        remaining = len(state["steps"]) - len(results_by_id(state))
        return "worker" if remaining > 0 else "solver"

    graph = StateGraph(PlanState)
    graph.add_node("planner", make_planner(json_model or model, settings, require_tools=True))
    graph.add_node("review", make_review(settings, require_tools=True))
    graph.add_node("worker", worker)
    graph.add_node("solver", solver)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", after_planning, ["review", END])
    graph.add_conditional_edges("review", after_review, ["worker", END])
    graph.add_conditional_edges("worker", after_worker, ["worker", "solver"])
    graph.add_edge("solver", END)
    return graph.compile(checkpointer=checkpointer, name="rewoo")
