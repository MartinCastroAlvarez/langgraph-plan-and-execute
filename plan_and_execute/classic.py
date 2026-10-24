"""Variant 1, classic plan-and-execute: planner -> (executor -> replanner)* -> answer.

The executor is a ReAct tool agent built with ``langchain.agents.create_agent`` and
runs one step at a time. After every step the replanner sees the results and either
finishes, continues, inserts recovery steps or replaces the rest of the plan. The
budget (steps, replans, tool calls) is checked by the graph, not by the model.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from plan_and_execute.common import make_finalizer
from plan_and_execute.common import make_planner
from plan_and_execute.common import make_review
from plan_and_execute.common import model_text
from plan_and_execute.config import Settings
from plan_and_execute.plan import Plan
from plan_and_execute.plan import PlanError
from plan_and_execute.plan import Step
from plan_and_execute.plan import parse_decision
from plan_and_execute.prompts import EXECUTOR
from plan_and_execute.prompts import REPLANNER
from plan_and_execute.prompts import data_block
from plan_and_execute.state import PlanState
from plan_and_execute.state import StepResult
from plan_and_execute.state import emit
from plan_and_execute.state import plan_of
from plan_and_execute.state import results_by_id
from plan_and_execute.state import short as _short
from plan_and_execute.tools import Toolbox

__all__ = ["build_classic", "build_executor"]

MAX_OUTPUT_CHARS = 1000
MAX_ERROR_CHARS = 200


def build_executor(
    model: BaseChatModel, toolbox: Toolbox, *, tool_limit: int
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """The ReAct executor: ``create_agent`` with the four tools and a per-step tool cap."""
    return create_agent(
        model,
        toolbox.langchain_tools(),
        system_prompt=EXECUTOR,
        middleware=[ToolCallLimitMiddleware(run_limit=tool_limit, exit_behavior="end")],
        name="executor",
    )


def _step_result(step: Step, messages: list[Any]) -> StepResult:
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    replies = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
    answer = model_text(replies[-1]) if replies else ""
    if not answer and tool_messages:
        answer = tool_messages[-1].text
    succeeded = [m for m in tool_messages if m.status != "error"]
    ok = bool(answer) and not answer.upper().startswith("FAILED")
    if step.tool is not None:
        ok = ok and bool(succeeded) and tool_messages[-1].status != "error"
    calls = [c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    return {
        "id": step.id,
        "round": 1,
        "tool": calls[-1]["name"] if calls else None,
        "input": str(calls[-1]["args"].get("input", "")) if calls else "",
        "output": (answer or "FAILED: no answer")[:MAX_OUTPUT_CHARS],
        "ok": ok,
        "tool_calls": len(tool_messages),
    }


def build_classic(
    model: BaseChatModel,
    toolbox: Toolbox,
    settings: Settings,
    *,
    json_model: BaseChatModel | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[PlanState, None, PlanState, PlanState]:
    """Compile the classic graph. ``json_model`` (planner, replanner) defaults to ``model``."""
    budget = settings.budget
    planning_model = json_model or model

    async def execute(state: PlanState) -> dict[str, Any]:
        tools_left = budget.max_tool_calls - state.get("tool_calls", 0)
        if state.get("steps_run", 0) >= budget.max_steps:
            return {"status": "budget", "stop_reason": f"max_steps={budget.max_steps} reached"}
        if tools_left <= 0:
            return {
                "status": "budget",
                "stop_reason": f"max_tool_calls={budget.max_tool_calls} reached",
            }
        plan = plan_of(state)
        pending = list(state.get("pending", []))
        step = plan.by_id()[pending[0]]
        done = results_by_id(state)
        evidence = {f"#E{i}": done[i]["output"] for i in step.depends_on if i in done}
        emit({"event": "step", "id": step.id, "goal": step.goal})
        agent = build_executor(
            model, toolbox, tool_limit=min(budget.max_step_tool_calls, tools_left)
        )
        message = HumanMessage(
            "\n".join(
                [
                    data_block("task", state["task"]),
                    data_block("step", step.model_dump()),
                    data_block("results", evidence),
                ]
            )
        )
        try:
            out = await agent.ainvoke({"messages": [message]})
            result = _step_result(step, out["messages"])
        except GraphBubbleUp:
            raise
        except Exception as exc:  # a model/provider error fails the step, not the run
            result = {
                "id": step.id,
                "round": 1,
                "tool": step.tool,
                "input": step.input or "",
                "output": f"FAILED: {type(exc).__name__}: {str(exc)[:MAX_ERROR_CHARS]}",
                "ok": False,
                "tool_calls": 0,
            }
        emit(
            {
                "event": "step_done",
                "id": step.id,
                "ok": result["ok"],
                "output": _short(result["output"]),
            }
        )
        return {
            "results": [result],
            "pending": pending[1:],
            "steps_run": 1,
            "tool_calls": result["tool_calls"],
        }

    async def replan(state: PlanState) -> dict[str, Any]:
        plan = plan_of(state)
        done = results_by_id(state)
        last = state["results"][-1]
        replans = state.get("replans", 0)
        steps_view = [
            {**s.model_dump(), "status": _status(s.id, done, state.get("pending", []))}
            for s in plan.steps
        ]
        budget_left = {
            "steps": budget.max_steps - state.get("steps_run", 0),
            "replans": budget.max_replans - replans,
            "tool_calls": budget.max_tool_calls - state.get("tool_calls", 0),
        }
        prompt = "\n".join(
            [
                data_block("task", state["task"]),
                data_block("plan", steps_view),
                data_block("results", {str(i): _view(r) for i, r in done.items()}),
                data_block("last_result", _view(last) | {"id": last["id"]}),
                data_block("pending", state.get("pending", [])),
                data_block("budget_left", budget_left),
            ]
        )
        reply = await planning_model.ainvoke([SystemMessage(REPLANNER), HumanMessage(prompt)])
        try:
            decision = parse_decision(
                model_text(reply), allowed={"continue", "finish", "insert", "replace"}
            )
        except PlanError as exc:
            emit({"event": "decision_invalid", "error": str(exc)})
            return {"history": [{"event": "decision_invalid", "error": str(exc)}]}
        if decision.action == "finish":
            emit({"event": "finish"})
            return {"answer": decision.answer or "", "status": "done"}
        if decision.action == "continue":
            return {}
        if replans >= budget.max_replans:
            emit({"event": "replan_refused", "action": decision.action})
            return {"history": [{"event": "refused", "action": decision.action}]}
        kept = [s for s in plan.steps if s.id in done]
        rest = [s for s in plan.steps if s.id in set(state.get("pending", []))]
        new_steps = kept + decision.steps + (rest if decision.action == "insert" else [])
        try:
            new_plan = Plan(steps=new_steps)
        except ValueError as exc:  # pydantic ValidationError is a ValueError
            emit({"event": "decision_invalid", "error": str(exc)[:MAX_ERROR_CHARS]})
            return {"history": [{"event": "decision_invalid", "error": str(exc)[:200]}]}
        pending = [i for i in new_plan.order() if i not in done]
        emit({"event": decision.action, "steps": [s.id for s in decision.steps]})
        return {
            "steps": [s.model_dump() for s in new_plan.steps],
            "pending": pending,
            "replans": replans + 1,
            "history": [
                {
                    "event": decision.action,
                    "after": last["id"],
                    "steps": [s.id for s in decision.steps],
                    "reason": decision.reason,
                }
            ],
        }

    def after_planning(state: PlanState) -> str:
        return "review" if state.get("status") == "running" else END

    def after_review(state: PlanState) -> str:
        return "execute" if state.get("status") == "running" else END

    def after_execute(state: PlanState) -> str:
        return "finalize" if state.get("status") == "budget" else "replan"

    def after_replan(state: PlanState) -> str:
        if state.get("status") == "done":
            return END
        return "execute" if state.get("pending") else "finalize"

    graph = StateGraph(PlanState)
    graph.add_node("planner", make_planner(planning_model, settings, require_tools=False))
    graph.add_node("review", make_review(settings, require_tools=False))
    graph.add_node("execute", execute)
    graph.add_node("replan", replan)
    graph.add_node("finalize", make_finalizer())
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", after_planning, ["review", END])
    graph.add_conditional_edges("review", after_review, ["execute", END])
    graph.add_conditional_edges("execute", after_execute, ["replan", "finalize"])
    graph.add_conditional_edges("replan", after_replan, ["execute", "finalize", END])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, name="classic")


def _status(step_id: int, done: dict[int, StepResult], pending: list[int]) -> str:
    if step_id in done:
        return "done" if done[step_id]["ok"] else "failed"
    return "pending" if step_id in pending else "dropped"


def _view(result: StepResult) -> dict[str, Any]:
    return {"output": result["output"], "ok": result["ok"]}
