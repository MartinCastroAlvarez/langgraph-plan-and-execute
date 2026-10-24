"""Compare the three variants: model calls, tool calls and wall clock per task.

``python -m plan_and_execute.bench`` runs the real graphs, tools and notes search on
three bundled tasks. Only the model is replaced, by ``FakePlanModel`` sleeping
``--latency`` seconds per call; tools sleep ``--tool-latency`` seconds per call. That
makes the numbers about orchestration, not about one model's speed: how many round
trips each variant needs and how many of them it can overlap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

from plan_and_execute.config import VARIANTS
from plan_and_execute.config import Settings
from plan_and_execute.fake import FakePlanModel
from plan_and_execute.graphs import build_graph
from plan_and_execute.graphs import make_toolbox
from plan_and_execute.progress import CallCounter

__all__ = ["TASKS", "Row", "main", "run_benchmark"]

TASKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "launch+flight (5 steps, 2 chains)": (
        "What weekday is the Project Atlas launch, and how many miles is the Berlin to "
        "Lisbon flight?",
        ("2026-12-18", "Friday", "1435.37 mi"),
    ),
    "per diem (3 steps)": (
        "What do 4 days of Lisbon per diem cost for the Atlas team?",
        ("2040",),
    ),
    "freezer (step fails)": (
        "How cold is the sample freezer in Fahrenheit?",
        ("-112 °F",),
    ),
}


@dataclass(frozen=True, slots=True)
class Row:
    task: str
    variant: str
    llm_calls: int
    tool_calls: int
    replans: int
    seconds: float
    correct: bool
    status: str


async def _once(variant: str, task: str, settings: Settings) -> tuple[dict[str, Any], int, float]:
    model = FakePlanModel(latency=settings.fake_latency)
    graph = build_graph(variant, settings, model=model, toolbox=make_toolbox(settings))
    counter = CallCounter()
    config = settings.run_config()
    config["callbacks"] = [counter]
    started = time.perf_counter()
    final = await graph.ainvoke({"task": task}, config)
    return final, counter.total, time.perf_counter() - started


async def run_benchmark(
    *, latency: float, tool_latency: float, repeats: int, variants: Sequence[str] = VARIANTS
) -> list[Row]:
    settings = Settings(
        provider="fake", fake_latency=latency, tool_latency=tool_latency, today=date(2026, 10, 24)
    )
    rows = []
    for label, (task, expected) in TASKS.items():
        for variant in variants:
            times = []
            for _ in range(repeats):
                final, calls, seconds = await _once(variant, task, settings)
                times.append(seconds)
            answer = final.get("answer", "")
            rows.append(
                Row(
                    task=label,
                    variant=variant,
                    llm_calls=calls,
                    tool_calls=final.get("tool_calls", 0),
                    replans=final.get("replans", 0),
                    seconds=round(statistics.median(times), 3),
                    correct=all(e in answer for e in expected),
                    status=final.get("status", ""),
                )
            )
    return rows


def table(rows: Sequence[Row]) -> str:
    lines = [
        "| Task | Variant | LLM calls | Tool calls | Replans | Wall clock (s) | Correct |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.task} | {r.variant} | {r.llm_calls} | {r.tool_calls} | {r.replans} | "
            f"{r.seconds:.3f} | {'yes' if r.correct else 'no'} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m plan_and_execute.bench", description=__doc__)
    p.add_argument("--latency", type=float, default=0.25, help="seconds per model call")
    p.add_argument("--tool-latency", type=float, default=0.1, help="seconds per tool call")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--json", type=Path, help="also write the results to this file")
    args = p.parse_args(argv)
    rows = asyncio.run(
        run_benchmark(latency=args.latency, tool_latency=args.tool_latency, repeats=args.repeats)
    )
    print(table(rows))
    if args.json:
        meta = {
            "date": date.today().isoformat(),
            "python": platform.python_version(),
            "langgraph": version("langgraph"),
            "langchain": version("langchain"),
            "latency": args.latency,
            "tool_latency": args.tool_latency,
            "repeats": args.repeats,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
