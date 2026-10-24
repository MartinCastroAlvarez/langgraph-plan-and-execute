"""The plan: Pydantic schemas, tolerant JSON parsing, validation and ``#E`` placeholders.

All three variants share one plan format, so the same planner prompt, parser and
human review work for each of them::

    {"steps": [
      {"id": 1, "goal": "Find the kickoff date", "tool": "search_notes",
       "input": "Project Atlas kickoff date", "success": "an ISO date", "depends_on": []},
      {"id": 2, "goal": "Add 45 days", "tool": "date_math", "input": "#E1 + 45 days",
       "success": "a date and weekday", "depends_on": [1]}
    ]}

``#E<n>`` in an ``input`` stands for the result of step ``n`` (ReWOO's evidence
variables). The classic executor reads it as a hint; ReWOO and the LLM-compiler
substitute it literally and never ask a model in between.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_validator

__all__ = [
    "TOOL_NAMES",
    "Decision",
    "Plan",
    "PlanError",
    "Step",
    "ToolName",
    "evidence_value",
    "extract_json",
    "parse_decision",
    "parse_plan",
    "placeholders",
    "substitute",
]

ToolName = Literal["calculator", "search_notes", "convert_units", "date_math"]
TOOL_NAMES: tuple[str, ...] = ("calculator", "search_notes", "convert_units", "date_math")

PLACEHOLDER = re.compile(r"#E(\d+)\b")
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_ID_PREFIX = re.compile(r"^\[[^\]]{1,80}\]\s*")
MAX_EVIDENCE_CHARS = 300


class PlanError(ValueError):
    """The model (or a human) produced a plan or decision that cannot be used."""


class Step(BaseModel):
    """One step: what to achieve, which tool should help, and how to tell it worked."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: int = Field(ge=1, le=99)
    goal: str = Field(min_length=1, max_length=300)
    tool: ToolName | None = None
    input: str | None = Field(default=None, max_length=300)
    success: str = Field(default="", max_length=300)
    depends_on: list[int] = Field(default_factory=list, max_length=10)

    @field_validator("tool", mode="before")
    @classmethod
    def _tool(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower()
            return None if value in {"", "none", "null", "llm"} else value
        return value

    @model_validator(mode="after")
    def _references(self) -> Step:
        refs = placeholders(self.input or "")
        deps = sorted(set(self.depends_on) | refs)
        if self.id in deps:
            raise ValueError(f"step {self.id} depends on itself")
        self.depends_on = deps
        return self


class Plan(BaseModel):
    """An ordered list of steps whose dependencies form a DAG."""

    model_config = ConfigDict(extra="ignore")

    steps: list[Step] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _graph(self) -> Plan:
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError(f"step ids must be unique, got {ids}")
        known = set(ids)
        for step in self.steps:
            missing = [d for d in step.depends_on if d not in known]
            if missing:
                raise ValueError(f"step {step.id} depends on unknown steps {missing}")
        self.waves()  # raises on a cycle
        return self

    def by_id(self) -> dict[int, Step]:
        return {s.id: s for s in self.steps}

    def waves(self, done: Iterable[int] = ()) -> list[list[int]]:
        """Group the steps into dependency levels: everything in a wave can run in parallel."""
        finished = set(done)
        remaining = {s.id: set(s.depends_on) for s in self.steps if s.id not in finished}
        order = [s.id for s in self.steps]
        waves: list[list[int]] = []
        while remaining:
            ready = [i for i in order if i in remaining and remaining[i] <= finished]
            if not ready:
                raise ValueError(f"dependency cycle among steps {sorted(remaining)}")
            waves.append(ready)
            finished.update(ready)
            for i in ready:
                del remaining[i]
        return waves

    def order(self) -> list[int]:
        """A topological order that keeps the planner's order wherever it can."""
        return [i for wave in self.waves() for i in wave]

    def check(self, *, max_steps: int, require_tools: bool = False) -> Plan:
        """Budget- and variant-specific checks the schema alone can't express."""
        if len(self.steps) > max_steps:
            raise PlanError(f"the plan has {len(self.steps)} steps; the budget allows {max_steps}")
        if require_tools:
            missing = [s.id for s in self.steps if s.tool is None or s.input is None]
            if missing:
                raise PlanError(f"steps {missing} need a tool and an input in this variant")
        return self


class Decision(BaseModel):
    """What the replanner (classic) or the joiner (compiler) wants to do next."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    action: Literal["continue", "finish", "replace", "insert", "replan"]
    answer: str | None = Field(default=None, max_length=4000)
    steps: list[Step] = Field(default_factory=list, max_length=20)
    reason: str = Field(default="", max_length=500)

    @field_validator("action", mode="before")
    @classmethod
    def _action(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _payload(self) -> Decision:
        if self.action == "finish" and not self.answer:
            raise ValueError("a finish decision needs an answer")
        if self.action in {"replace", "insert"} and not self.steps:
            raise ValueError(f"a {self.action} decision needs steps")
        return self


def extract_json(text: str) -> Any:
    """The first JSON object or array in ``text``: tolerates prose, code fences and <think>."""
    cleaned = _THINK.sub("", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict | list):
            return value
    raise PlanError("no JSON object in the model output")


def _errors(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        where = ".".join(str(p) for p in err["loc"]) or "plan"
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)


def parse_plan(text: str, *, max_steps: int, require_tools: bool = False) -> Plan:
    """Parse and validate a planner reply. Raises :class:`PlanError` with a short reason."""
    data = extract_json(text)
    if isinstance(data, list):
        data = {"steps": data}
    try:
        plan = Plan.model_validate(data)
    except ValidationError as exc:
        raise PlanError(_errors(exc)) from None
    return plan.check(max_steps=max_steps, require_tools=require_tools)


def parse_decision(text: str, *, allowed: Iterable[str]) -> Decision:
    allowed = set(allowed)
    data = extract_json(text)
    if not isinstance(data, dict):
        raise PlanError("a decision must be a JSON object")
    try:
        decision = Decision.model_validate(data)
    except ValidationError as exc:
        raise PlanError(_errors(exc)) from None
    if decision.action not in allowed:
        raise PlanError(f"action {decision.action!r} is not one of {sorted(allowed)}")
    return decision


def placeholders(text: str) -> set[int]:
    return {int(m) for m in PLACEHOLDER.findall(text)}


def evidence_value(output: str, *, numeric: bool = False) -> str:
    """The part of a step result that a placeholder stands for.

    The first line, without a ``[note-id]`` prefix and capped in length. Inside a
    calculator expression the placeholder becomes the first number of that line,
    because ``85 EUR per day * 6`` is not arithmetic.
    """
    lines = output.strip().splitlines()
    first = _ID_PREFIX.sub("", lines[0]).strip() if lines else ""
    first = first[:MAX_EVIDENCE_CHARS]
    if not numeric:
        return first
    match = _NUMBER.search(first)
    if match is None:
        raise PlanError(f"no number in the evidence {first!r}")
    return match.group(0).replace(",", "")


def substitute(text: str, results: Mapping[int, str], *, numeric: bool = False) -> str:
    """Replace every ``#E<n>`` with the evidence of step ``n``."""

    def repl(match: re.Match[str]) -> str:
        ref = int(match.group(1))
        if ref not in results:
            raise PlanError(f"#E{ref} has no result yet")
        return evidence_value(results[ref], numeric=numeric)

    return PLACEHOLDER.sub(repl, text)
