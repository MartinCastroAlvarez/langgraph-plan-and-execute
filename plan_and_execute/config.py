"""Settings and the plan-level budget, read from ``PLAN_EXECUTE_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from typing import Literal

from langchain_core.runnables import RunnableConfig

__all__ = ["DEFAULT_MODELS", "PROVIDERS", "VARIANTS", "Budget", "Provider", "Settings", "Variant"]

Provider = Literal["ollama", "anthropic", "fake"]
PROVIDERS: tuple[Provider, ...] = ("ollama", "anthropic", "fake")
Variant = Literal["classic", "rewoo", "compiler"]
VARIANTS: tuple[Variant, ...] = ("classic", "rewoo", "compiler")

DEFAULT_MODELS: dict[Provider, str] = {
    "ollama": "qwen3:4b",
    "anthropic": "claude-sonnet-5-5",
    "fake": "fake-planner",
}


def _int(env: Mapping[str, str], key: str, default: int, *, minimum: int = 0) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    value = float(raw)
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Budget:
    """Hard limits for one task. Hitting any of them ends the run with a partial answer."""

    max_steps: int = 8
    """Steps in a plan, and steps the classic executor may run in total."""
    max_replans: int = 3
    """Plan edits (replace / insert) by the replanner, or re-plans by the joiner."""
    max_tool_calls: int = 12
    """Tool calls across the whole task."""
    max_step_tool_calls: int = 3
    """Tool calls the classic executor may make for one step."""

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_tool_calls", "max_step_tool_calls"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_replans < 0:
            raise ValueError("max_replans must be >= 0")


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the three graphs and the CLI can be tuned with."""

    provider: Provider = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    ollama_host: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False
    temperature: float = 0.0
    budget: Budget = field(default_factory=Budget)
    approve_plan: bool = False
    """Pause with ``interrupt()`` after planning so a human can approve, edit or reject."""
    notes_dir: str | None = None
    today: date | None = None
    """Fixed "today" for the date tool (tests, reproducible demos); ``None`` = the real date."""
    fake_latency: float = 0.0
    """Seconds per model call for the ``fake`` provider."""
    tool_latency: float = 0.0
    """Simulated seconds per tool call (any provider), to make parallel scheduling visible."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        provider = e.get("PLAN_EXECUTE_PROVIDER", "ollama").strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"PLAN_EXECUTE_PROVIDER must be one of {PROVIDERS}")
        p: Provider = provider
        today = e.get("PLAN_EXECUTE_TODAY") or None
        return cls(
            provider=p,
            model=e.get("PLAN_EXECUTE_MODEL") or DEFAULT_MODELS[p],
            ollama_host=e.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
            allow_remote_ollama=e.get("PLAN_EXECUTE_ALLOW_REMOTE_OLLAMA", "") == "1",
            temperature=_float(e, "PLAN_EXECUTE_TEMPERATURE", 0.0),
            budget=Budget(
                max_steps=_int(e, "PLAN_EXECUTE_MAX_STEPS", 8, minimum=1),
                max_replans=_int(e, "PLAN_EXECUTE_MAX_REPLANS", 3),
                max_tool_calls=_int(e, "PLAN_EXECUTE_MAX_TOOL_CALLS", 12, minimum=1),
                max_step_tool_calls=_int(e, "PLAN_EXECUTE_MAX_STEP_TOOL_CALLS", 3, minimum=1),
            ),
            approve_plan=e.get("PLAN_EXECUTE_APPROVE", "") == "1",
            notes_dir=e.get("PLAN_EXECUTE_NOTES_DIR") or None,
            today=date.fromisoformat(today) if today else None,
            fake_latency=_float(e, "PLAN_EXECUTE_FAKE_LATENCY", 0.0),
            tool_latency=_float(e, "PLAN_EXECUTE_TOOL_LATENCY", 0.0),
        )

    def run_config(self, thread_id: str | None = None) -> RunnableConfig:
        """The LangGraph ``config`` for one run, with a recursion limit sized to the budget."""
        b = self.budget
        config: RunnableConfig = {
            "recursion_limit": 20 + 4 * (b.max_steps + b.max_replans) + 2 * b.max_tool_calls
        }
        if thread_id is not None:
            config["configurable"] = {"thread_id": thread_id}
        return config
