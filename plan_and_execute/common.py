"""Nodes every variant shares: the planner, the human plan review and the finalizer."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any
from typing import Literal
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import model_validator

from plan_and_execute.config import Settings
from plan_and_execute.plan import Plan
from plan_and_execute.plan import PlanError
from plan_and_execute.plan import Step
from plan_and_execute.plan import evidence_value
from plan_and_execute.plan import parse_plan
from plan_and_execute.prompts import data_block
from plan_and_execute.prompts import planner_prompt
from plan_and_execute.state import PlanState
from plan_and_execute.state import StepResult
from plan_and_execute.state import emit
from plan_and_execute.state import plan_of
from plan_and_execute.state import results_by_id

__all__ = [
    "ReviewDecision",
    "compose_answer",
    "make_finalizer",
    "make_planner",
    "make_review",
    "model_text",
]


class Node(Protocol):
    """An async LangGraph node over :class:`PlanState`."""

    def __call__(self, state: PlanState) -> Awaitable[dict[str, Any]]: ...


MAX_REVIEW_ATTEMPTS = 3
MAX_ANSWER_CHARS = 4000


def model_text(message: BaseMessage) -> str:
    """The text of a model reply, without a ``<think>`` block (qwen3, deepseek-r1)."""
    text = str(message.text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def _round(state: PlanState) -> int:
    return state.get("round", 1)


def make_planner(model: BaseChatModel, settings: Settings, *, require_tools: bool) -> Node:
    """Ask for a plan, validate it, and retry once with the validation error."""
    max_steps = settings.budget.max_steps
    system = SystemMessage(planner_prompt(max_steps=max_steps, require_tools=require_tools))

    async def planner(state: PlanState) -> dict[str, Any]:
        round_ = _round(state)
        content = data_block("task", state["task"])
        if round_ > 1:
            previous = [
                {"id": r["id"], "input": r["input"], "output": r["output"], "ok": r["ok"]}
                for r in results_by_id(state, round_ - 1).values()
            ]
            content += "\n" + data_block("previous_results", previous)
            content += "\n" + data_block("replan_reason", state.get("replan_reason", ""))
            content += "\nWrite a new plan that fixes what failed. Ids start again at 1."
        messages: list[BaseMessage] = [system, HumanMessage(content)]
        error = ""
        for _attempt in range(2):
            reply = await model.ainvoke(messages)
            try:
                plan = parse_plan(
                    model_text(reply), max_steps=max_steps, require_tools=require_tools
                )
            except PlanError as exc:
                error = str(exc)
                emit({"event": "plan_invalid", "error": error})
                messages += [
                    AIMessage(model_text(reply)[:4000]),
                    HumanMessage(f"That plan is invalid: {error}. Reply with corrected JSON only."),
                ]
                continue
            steps = [s.model_dump() for s in plan.steps]
            emit({"event": "plan", "round": round_, "steps": [_brief(s) for s in plan.steps]})
            return {
                "steps": steps,
                "pending": plan.order(),
                "status": "running",
                "round": round_,
                "replans": state.get("replans", 0),
                "history": [{"event": "planned", "round": round_, "steps": len(steps)}],
            }
        return {
            "status": "failed",
            "stop_reason": f"invalid plan: {error}",
            "answer": f"I could not make a valid plan for this task ({error}).",
            "history": [{"event": "plan_failed", "round": round_, "error": error}],
        }

    return planner


def _brief(step: Step) -> str:
    deps = f" <- {step.depends_on}" if step.depends_on else ""
    return f"{step.id}. {step.tool or '-'}({step.input or step.goal}){deps}"


class ReviewDecision(BaseModel):
    """What a human answers to the plan-review interrupt (``Command(resume=...)``)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["approve", "edit", "reject"]
    steps: list[Step] | None = Field(default=None, max_length=20)
    reason: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _edit(self) -> ReviewDecision:
        if self.action == "edit":
            if not self.steps:
                raise ValueError("an edit needs the full list of steps")
            try:
                Plan(steps=self.steps)  # unique ids, known dependencies, no cycle
            except ValidationError as exc:
                raise ValueError(exc.errors()[0]["msg"]) from None
        return self


def make_review(settings: Settings, *, require_tools: bool) -> Node:
    """Pause for a human when ``approve_plan`` is on: approve, edit or reject the plan.

    ``interrupt(..., response_schema=ReviewDecision)`` validates the resume value: a
    malformed answer raises ``ValidationError`` to the caller and nothing is saved, so
    the same interrupt can simply be answered again. An edit that is well-formed but
    breaks the budget or the variant's rules gets a second interrupt with the error.
    """

    async def review(state: PlanState) -> dict[str, Any]:
        if not settings.approve_plan or _round(state) > 1 or state.get("status") != "running":
            return {}
        error = ""
        for _attempt in range(MAX_REVIEW_ATTEMPTS):
            payload = {
                "kind": "plan_review",
                "task": state["task"],
                "steps": state["steps"],
                "error": error,
                "answer_with": ReviewDecision.model_json_schema()["properties"],
            }
            decision: ReviewDecision = interrupt(payload, response_schema=ReviewDecision)
            if decision.action == "approve":
                emit({"event": "approved"})
                return {"history": [{"event": "approved"}]}
            if decision.action == "reject":
                emit({"event": "rejected", "reason": decision.reason})
                return {
                    "status": "rejected",
                    "stop_reason": "plan rejected by the reviewer",
                    "answer": f"The plan was rejected. {decision.reason}".strip(),
                    "history": [{"event": "rejected", "reason": decision.reason}],
                }
            try:
                plan = Plan(steps=decision.steps or []).check(
                    max_steps=settings.budget.max_steps, require_tools=require_tools
                )
            except PlanError as exc:
                error = str(exc)
                continue
            emit({"event": "edited", "steps": [_brief(s) for s in plan.steps]})
            return {
                "steps": [s.model_dump() for s in plan.steps],
                "pending": plan.order(),
                "history": [{"event": "edited", "steps": len(plan.steps)}],
            }
        return {
            "status": "rejected",
            "stop_reason": "no valid plan after review",
            "answer": f"The plan was not approved ({error}).",
            "history": [{"event": "rejected", "reason": error}],
        }

    return review


def compose_answer(state: PlanState, results: dict[int, StepResult] | None = None) -> str:
    """A deterministic answer from whatever the steps found (no model call)."""
    found = results if results is not None else results_by_id(state, _round(state))
    if not found:
        return "No step was completed."
    goals = {s.id: s.goal for s in plan_of(state).steps} if state.get("steps") else {}
    good = [
        f"{goals.get(i, f'step {i}')}: {evidence_value(r['output'])}"
        for i, r in sorted(found.items())
        if r["ok"]
    ]
    bad = [
        f"{goals.get(i, f'step {i}')} ({evidence_value(r['output'])})"
        for i, r in sorted(found.items())
        if not r["ok"]
    ]
    text = ("Found: " + "; ".join(good) + ".") if good else "Nothing was found."
    if bad:
        text += " Could not complete: " + "; ".join(bad) + "."
    return text[:MAX_ANSWER_CHARS]


def make_finalizer() -> Node:
    """End without a model call: on a budget stop, or when nothing wrote an answer."""

    async def finalize(state: PlanState) -> dict[str, Any]:
        answer = compose_answer(state)
        status = state.get("status", "running")
        if status == "budget":
            answer = f"Stopped early ({state.get('stop_reason', 'budget')}). {answer}"
        else:
            status = "done"
        emit({"event": "finalized", "status": status})
        return {"answer": answer, "status": status}

    return finalize
