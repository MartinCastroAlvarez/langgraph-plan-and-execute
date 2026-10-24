"""A deterministic, role-aware fake chat model with simulated latency.

``FakePlanModel`` is a real LangChain ``BaseChatModel``, so the graphs call it exactly
as they call ``ChatOllama``. It reads the ``Role: <name>.`` line of the system prompt
and answers like that node would:

* **planner** - the plan of the first matching :data:`SCENARIOS` entry (all keywords in
  the task), or a one-step ``search_notes`` plan for any other task;
* **executor** - one tool call built from the step's ``input`` (``#E<n>`` filled from
  the results it was given), then the tool output as its answer;
* **replanner** - inserts the scenario's recovery steps after a failed step, continues
  while steps remain, then finishes;
* **solver** / **joiner** - an answer from the scenario template, or a re-plan request
  when the joiner sees a failure the scenario knows how to fix.

``script={"role": [reply, ...]}`` overrides any role with canned replies (strings,
``AIMessage``s, or exceptions to raise), consumed in order. Every call is logged by role and sleeps
``latency`` seconds, which makes the benchmark's wall-clock numbers meaningful.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field
from pydantic import PrivateAttr

from plan_and_execute.plan import PlanError
from plan_and_execute.plan import evidence_value
from plan_and_execute.plan import substitute
from plan_and_execute.prompts import parse_blocks
from plan_and_execute.prompts import role_of

__all__ = ["SCENARIOS", "FakePlanModel", "Scenario"]


def _step(
    id: int, goal: str, tool: str, input: str, success: str, deps: Sequence[int] = ()
) -> dict[str, Any]:
    return {
        "id": id,
        "goal": goal,
        "tool": tool,
        "input": input,
        "success": success,
        "depends_on": list(deps),
    }


@dataclass(frozen=True)
class Scenario:
    """A canned plan for a known task, with optional recovery and re-plan behaviour."""

    name: str
    keywords: tuple[str, ...]
    plan: list[dict[str, Any]]
    answers: tuple[str, ...]
    """Answer templates with ``#E<n>``; the first whose steps all succeeded is used."""
    recovery: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    """Classic: steps the replanner inserts when step ``n`` fails."""
    replan: list[dict[str, Any]] | None = None
    """Compiler: the planner's second plan, after the joiner asks for one."""

    def matches(self, task: str) -> bool:
        lowered = task.lower()
        return all(k in lowered for k in self.keywords)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="launch-and-flight",
        keywords=("atlas", "launch", "miles"),
        plan=[
            _step(
                1,
                "Find the Atlas kickoff date",
                "search_notes",
                "Project Atlas kickoff date",
                "an ISO date",
            ),
            _step(
                2,
                "Find how long after kickoff Atlas launches",
                "search_notes",
                "Project Atlas launch after kickoff",
                "a number of days",
            ),
            _step(
                3,
                "Compute the launch date and weekday",
                "date_math",
                "#E1 + #E2",
                "a date with its weekday",
                [1, 2],
            ),
            _step(
                4,
                "Find the Berlin to Lisbon flight distance",
                "search_notes",
                "Berlin to Lisbon flight distance",
                "a distance in km",
            ),
            _step(
                5,
                "Convert the distance to miles",
                "convert_units",
                "#E4 to mi",
                "a distance in miles",
                [4],
            ),
        ],
        answers=("Project Atlas launches on #E3, and the Berlin to Lisbon flight is #E5.",),
    ),
    Scenario(
        name="per-diem",
        keywords=("per diem", "atlas"),
        plan=[
            _step(
                1,
                "Find the Lisbon per diem",
                "search_notes",
                "travel per diem Lisbon",
                "an amount per day",
            ),
            _step(
                2,
                "Find the Atlas team size",
                "search_notes",
                "Project Atlas team size",
                "a number of people",
            ),
            _step(
                3,
                "Multiply per diem x people x 4 days",
                "calculator",
                "#E1 * #E2 * 4",
                "a total in EUR",
                [1, 2],
            ),
        ],
        answers=("Four days of Lisbon per diems for the Atlas team come to #E3 EUR.",),
    ),
    Scenario(
        name="freezer-recovery",
        keywords=("freezer", "fahrenheit"),
        plan=[
            _step(
                1,
                "Find the freezer setpoint",
                "search_notes",
                "sample freezer setpoint",
                "a temperature in C",
            ),
            _step(
                2,
                "Convert the setpoint",
                "convert_units",
                "#E1 to degrees",
                "a temperature in F",
                [1],
            ),
        ],
        answers=(
            "#E1, which is #E3.",
            "#E1, which is #E2.",
        ),
        recovery={
            2: [
                _step(
                    3,
                    "Convert the setpoint to Fahrenheit",
                    "convert_units",
                    "#E1 to F",
                    "a temperature in F",
                    [1],
                )
            ],
        },
        replan=[
            _step(
                1,
                "Find the freezer setpoint",
                "search_notes",
                "sample freezer setpoint",
                "a temperature in C",
            ),
            _step(
                2,
                "Convert the setpoint to Fahrenheit",
                "convert_units",
                "#E1 to F",
                "a temperature in F",
                [1],
            ),
        ],
    ),
)


def _generic(task: str) -> Scenario:
    query = " ".join(task.split())[:300] or "notes"
    return Scenario(
        name="generic",
        keywords=(),
        plan=[_step(1, "Search the notes for the task", "search_notes", query, "a fact")],
        answers=("From the notes: #E1",),
    )


def _results(raw: str) -> dict[int, dict[str, Any]]:
    """``{"1": {"output": ..., "ok": ...}}`` or a list of ``{"id": ...}`` -> by id."""
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    if isinstance(data, list):
        return {int(r["id"]): r for r in data if isinstance(r, dict) and "id" in r}
    return {int(k): v for k, v in data.items() if isinstance(v, dict)}


def _render(scenario: Scenario, results: dict[int, dict[str, Any]]) -> str:
    ok = {i: str(r.get("output", r.get("result", ""))) for i, r in results.items() if r.get("ok")}
    for template in scenario.answers:
        try:
            return substitute(template, ok)
        except PlanError:
            continue
    found = "; ".join(evidence_value(v) for _, v in sorted(ok.items())) or "nothing"
    return f"I could only find part of the answer: {found}."


class FakePlanModel(BaseChatModel):
    """Role-aware, deterministic fake; see the module docstring."""

    latency: float = 0.0
    script: dict[str, list[Any]] = Field(default_factory=dict)
    scenarios: tuple[Scenario, ...] = SCENARIOS
    latency_for: Callable[[str], float] | None = None

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _calls: list[str] = PrivateAttr(default_factory=list)
    _in_flight: int = PrivateAttr(default=0)
    _peak: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake-planner"

    @property
    def calls(self) -> list[str]:
        """The role of every call, in start order."""
        return list(self._calls)

    @property
    def peak_concurrency(self) -> int:
        return self._peak

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
            self._in_flight = self._peak = 0

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The fake knows the four tools and decides itself when to call them.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        role, delay = self._enter(messages)
        try:
            time.sleep(delay)
            return self._result(role, messages)
        finally:
            self._exit()

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        role, delay = self._enter(messages)
        try:
            await asyncio.sleep(delay)
            return self._result(role, messages)
        finally:
            self._exit()

    def _enter(self, messages: list[BaseMessage]) -> tuple[str, float]:
        system = next((m.text for m in messages if isinstance(m, SystemMessage)), "")
        role = role_of(system)
        with self._lock:
            self._calls.append(role)
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
        return role, self.latency_for(role) if self.latency_for else self.latency

    def _exit(self) -> None:
        with self._lock:
            self._in_flight -= 1

    def _result(self, role: str, messages: list[BaseMessage]) -> ChatResult:
        with self._lock:
            queue = self.script.get(role)
            canned = queue.pop(0) if queue else None
        if isinstance(canned, BaseException):
            raise canned
        if canned is not None:
            message = canned if isinstance(canned, AIMessage) else AIMessage(content=str(canned))
        else:
            message = self._respond(role, messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _scenario(self, task: str) -> Scenario:
        return next((s for s in self.scenarios if s.matches(task)), None) or _generic(task)

    def _respond(self, role: str, messages: list[BaseMessage]) -> AIMessage:
        prompt = "\n".join(m.text for m in messages if isinstance(m, HumanMessage))
        data = parse_blocks(prompt)
        scenario = self._scenario(data.get("task", ""))
        if role == "planner":
            replanning = "previous_results" in data and scenario.replan is not None
            plan = scenario.replan if replanning else scenario.plan
            return AIMessage(content=json.dumps({"steps": plan}))
        if role == "executor":
            return self._execute(data, messages)
        if role == "replanner":
            return AIMessage(content=json.dumps(self._replan(scenario, data)))
        if role == "solver":
            evidence = {
                int(e["id"]): {"output": e["evidence"], "ok": not _is_error(e["evidence"])}
                for e in json.loads(data.get("evidence", "[]"))
            }
            return AIMessage(content=_render(scenario, evidence))
        if role == "joiner":
            results = _results(data.get("results", ""))
            failed = any(not r.get("ok") for r in results.values())
            if failed and scenario.replan is not None and data.get("replans_left", "0") != "0":
                return AIMessage(
                    content=json.dumps({"action": "replan", "reason": "a step failed"})
                )
            return AIMessage(
                content=json.dumps({"action": "finish", "answer": _render(scenario, results)})
            )
        return AIMessage(content="ok")

    @staticmethod
    def _execute(data: dict[str, str], messages: list[BaseMessage]) -> AIMessage:
        step = json.loads(data.get("step", "{}"))
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_messages:
            last = tool_messages[-1]
            prefix = "FAILED: " if last.status == "error" else ""
            return AIMessage(content=prefix + last.text)
        if not step.get("tool"):
            return AIMessage(content=f"No tool needed: {step.get('goal', '')}")
        raw = json.loads(data.get("results", "{}") or "{}")
        evidence = {int(k.lstrip("#E")): str(v) for k, v in raw.items()}
        try:
            text = substitute(
                step.get("input") or step.get("goal", ""),
                evidence,
                numeric=step["tool"] == "calculator",
            )
        except PlanError as exc:
            return AIMessage(content=f"FAILED: {exc}")
        call = {"name": step["tool"], "args": {"input": text}, "id": f"call-{step['id']}"}
        return AIMessage(content="", tool_calls=[call])

    @staticmethod
    def _replan(scenario: Scenario, data: dict[str, str]) -> dict[str, Any]:
        last = json.loads(data.get("last_result", "{}"))
        plan_ids = {s["id"] for s in json.loads(data.get("plan", "[]"))}
        pending = json.loads(data.get("pending", "[]"))
        recovery = scenario.recovery.get(int(last.get("id", 0)), [])
        if not last.get("ok", True) and recovery and not {s["id"] for s in recovery} & plan_ids:
            return {"action": "insert", "steps": recovery, "reason": "the last step failed"}
        if pending:
            return {"action": "continue"}
        return {"action": "finish", "answer": _render(scenario, _results(data.get("results", "")))}


def _is_error(text: str) -> bool:
    return text.startswith(("error:", "skipped:", "not run"))
