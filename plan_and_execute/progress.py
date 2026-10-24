"""Stream progress events while a graph runs, and count model calls.

Nodes call :func:`~plan_and_execute.state.emit` (LangGraph's ``get_stream_writer``).
:func:`stream_run` streams ``custom`` and ``values`` and yields a :class:`Progress` per
event, then the final state. :class:`CallCounter` is a callback handler that counts
chat-model calls by role for any provider, which is how the CLI and the benchmark
report "LLM calls per task".
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from plan_and_execute.prompts import role_of

__all__ = ["CallCounter", "Progress", "format_progress", "stream_run"]


class CallCounter(BaseCallbackHandler):
    """Counts chat-model calls by the ``Role:`` of their system prompt."""

    def __init__(self) -> None:
        self.by_role: Counter[str] = Counter()
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return sum(self.by_role.values())

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        for batch in messages:
            system = next((m.text for m in batch if isinstance(m, SystemMessage)), "")
            with self._lock:
                self.by_role[role_of(system) or "?"] += 1


@dataclass(frozen=True, slots=True)
class Progress:
    seconds: float
    event: dict[str, Any]


async def stream_run(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    inputs: Any,
    config: RunnableConfig | None = None,
) -> AsyncIterator[Progress | dict[str, Any]]:
    """Yield a :class:`Progress` per custom event, then the final state (a ``dict``)."""
    started = time.perf_counter()
    final: dict[str, Any] = {}
    async for namespace, mode, data in graph.astream(
        inputs, config, stream_mode=["custom", "values"], subgraphs=True
    ):
        if mode == "custom" and isinstance(data, dict):
            yield Progress(round(time.perf_counter() - started, 3), data)
        elif mode == "values" and not namespace and isinstance(data, dict):
            final = data
    yield final


def format_progress(p: Progress) -> str:
    event = dict(p.event)
    kind = event.pop("event", "?")
    detail = " ".join(f"{k}={v!r}" for k, v in event.items())
    return f"[{p.seconds:7.3f}s] {kind:<14} {detail}".rstrip()
