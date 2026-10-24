"""One entry point for the three variants."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from plan_and_execute.classic import build_classic
from plan_and_execute.compiler import build_compiler
from plan_and_execute.config import VARIANTS
from plan_and_execute.config import Settings
from plan_and_execute.llm import make_model
from plan_and_execute.notes import Notes
from plan_and_execute.rewoo import build_rewoo
from plan_and_execute.tools import Toolbox

__all__ = ["build_graph", "make_toolbox"]


def make_toolbox(settings: Settings) -> Toolbox:
    return Toolbox(
        notes=Notes.from_dir(settings.notes_dir),
        today=settings.today,
        latency=settings.tool_latency,
    )


def build_graph(
    variant: str,
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    json_model: BaseChatModel | None = None,
    toolbox: Toolbox | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile ``classic``, ``rewoo`` or ``compiler``.

    Without ``model`` the provider in ``settings`` is used; ``json_model`` (planner,
    replanner, joiner) then gets ollama's JSON mode. A given ``model`` serves both roles.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if model is None:
        model = make_model(settings)
        json_model = json_model or make_model(settings, json_mode=True)
    tools = toolbox or make_toolbox(settings)
    builders = {"classic": build_classic, "rewoo": build_rewoo, "compiler": build_compiler}
    return builders[variant](
        model, tools, settings, json_model=json_model, checkpointer=checkpointer
    )
