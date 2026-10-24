"""``python -m plan_and_execute "<task>"``: plan, (approve,) execute, answer.

python -m plan_and_execute "What weekday is the Project Atlas launch?"          # classic, ollama
python -m plan_and_execute --variant rewoo "..."                                 # or compiler
python -m plan_and_execute --provider fake --variant all "..."                   # compare, offline
python -m plan_and_execute --approve "..."                                       # review the plan
python -m plan_and_execute --approve --db runs.sqlite --thread t1 "..."          # pause durably
python -m plan_and_execute --db runs.sqlite --thread t1 --resume approve         # ...resume later
python -m plan_and_execute --mermaid compiler                                    # print a graph
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any
from typing import TextIO

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import ValidationError

from plan_and_execute.classic import build_executor
from plan_and_execute.config import DEFAULT_MODELS
from plan_and_execute.config import PROVIDERS
from plan_and_execute.config import VARIANTS
from plan_and_execute.config import Settings
from plan_and_execute.fake import FakePlanModel
from plan_and_execute.graphs import build_graph
from plan_and_execute.graphs import make_toolbox
from plan_and_execute.llm import ModelConfigError
from plan_and_execute.progress import CallCounter
from plan_and_execute.progress import Progress
from plan_and_execute.progress import format_progress
from plan_and_execute.progress import stream_run

__all__ = ["main"]

MAX_TASK_CHARS = 4000
MAX_PLAN_FILE_BYTES = 100_000


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m plan_and_execute", description=(__doc__ or "").split("\n")[0]
    )
    p.add_argument("task", nargs="?", help="the task to plan and execute")
    p.add_argument("--variant", choices=(*VARIANTS, "all"), default="classic")
    p.add_argument("--provider", choices=PROVIDERS)
    p.add_argument("--model", help="model id for the provider")
    p.add_argument("--latency", type=float, help="seconds per call for --provider fake")
    p.add_argument("--tool-latency", type=float, help="simulated seconds per tool call")
    p.add_argument("--today", type=date.fromisoformat, help="fixed date for the date tool")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-replans", type=int)
    p.add_argument("--max-tool-calls", type=int)
    p.add_argument("--approve", action="store_true", help="review the plan before it runs")
    p.add_argument("--db", help="SQLite checkpoint file (durable pause/resume)")
    p.add_argument("--thread", help="thread id for --db (default: a new uuid)")
    p.add_argument(
        "--resume", metavar="approve|reject|edit:FILE", help="answer a paused plan review"
    )
    p.add_argument("--mermaid", choices=(*VARIANTS, "executor"), help="print a graph and exit")
    p.add_argument("--json", action="store_true", help="print the final state as JSON")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    return p


def _settings(args: argparse.Namespace) -> Settings:
    base = Settings.from_env()
    changes: dict[str, Any] = {}
    if args.provider:
        changes["provider"] = args.provider
        if not args.model:
            changes["model"] = DEFAULT_MODELS[args.provider]
    if args.model:
        changes["model"] = args.model
    if args.latency is not None:
        changes["fake_latency"] = args.latency
    if args.tool_latency is not None:
        changes["tool_latency"] = args.tool_latency
    if args.today is not None:
        changes["today"] = args.today
    budget = {
        k: v
        for k, v in {
            "max_steps": args.max_steps,
            "max_replans": args.max_replans,
            "max_tool_calls": args.max_tool_calls,
        }.items()
        if v is not None
    }
    if budget:
        changes["budget"] = dataclasses.replace(base.budget, **budget)
    if args.approve or args.resume:
        changes["approve_plan"] = True
    return dataclasses.replace(base, **changes)


def _resume_value(spec: str) -> dict[str, Any]:
    if spec in {"approve", "reject"}:
        return {"action": spec}
    if spec.startswith("edit:"):
        path = Path(spec[len("edit:") :])
        if path.stat().st_size > MAX_PLAN_FILE_BYTES:
            raise ValueError(f"{path} is larger than {MAX_PLAN_FILE_BYTES} bytes")
        data = json.loads(path.read_text("utf-8"))
        steps = data["steps"] if isinstance(data, dict) else data
        return {"action": "edit", "steps": steps}
    raise ValueError("--resume takes approve, reject or edit:<plan.json>")


def _print_review(payload: dict[str, Any], out: TextIO) -> None:
    print("\nPlan awaiting review:", file=out)
    for step in payload["steps"]:
        deps = f"  (after {step['depends_on']})" if step["depends_on"] else ""
        hint = f"{step['tool']}({step['input']})" if step["tool"] else "-"
        print(f"  {step['id']}. {step['goal']}  ->  {hint}{deps}", file=out)
    if payload.get("error"):
        print(f"  ! {payload['error']}", file=out)


async def _drive(
    graph: Any, inputs: Any, config: RunnableConfig, args: argparse.Namespace, out: TextIO
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Stream one run; return the final state and the pending review payload, if any."""
    while True:
        final: dict[str, Any] = {}
        async for item in stream_run(graph, inputs, config):
            if isinstance(item, Progress):
                if not args.quiet:
                    print(format_progress(item), file=out)
            else:
                final = item
        snapshot = await graph.aget_state(config) if graph.checkpointer else None
        pending = snapshot.interrupts if snapshot is not None else ()
        if not pending:
            return final, None
        payload: dict[str, Any] = pending[0].value
        _print_review(payload, out)
        if args.db:  # durable mode: stop here, resume from another process
            return final, payload
        answer = (await asyncio.to_thread(input, "approve / reject / edit:<plan.json> > ")).strip()
        answer = answer or "approve"
        try:
            inputs = Command(resume=_resume_value(answer))
        except (ValueError, OSError, KeyError) as exc:
            print(f"  ! {exc}", file=out)
            inputs = None  # re-show the same interrupt


async def _run_one(
    variant: str, task: str | None, settings: Settings, args: argparse.Namespace, out: TextIO
) -> dict[str, Any]:
    counter = CallCounter()
    thread = args.thread or str(uuid.uuid4())
    config = settings.run_config(thread)
    config["callbacks"] = [counter]
    inputs: Any = {"task": task}
    if args.resume:
        inputs = Command(resume=_resume_value(args.resume))
    started = time.perf_counter()
    if args.db:
        async with AsyncSqliteSaver.from_conn_string(args.db) as saver:
            graph = build_graph(variant, settings, checkpointer=saver)
            final, payload = await _drive(graph, inputs, config, args, out)
    else:
        memory = InMemorySaver() if settings.approve_plan else None
        graph = build_graph(variant, settings, checkpointer=memory)
        final, payload = await _drive(graph, inputs, config, args, out)
    seconds = time.perf_counter() - started
    if payload is not None:
        print(
            f"\nPaused for review (thread {thread}). Resume with:\n  python -m plan_and_execute "
            f"--variant {variant} --db {args.db} --thread {thread} --resume approve",
            file=out,
        )
        return {"variant": variant, "status": "paused", "thread": thread}
    return {
        "variant": variant,
        "status": final.get("status"),
        "answer": final.get("answer", ""),
        "llm_calls": counter.total,
        "llm_calls_by_role": dict(counter.by_role),
        "tool_calls": final.get("tool_calls", 0),
        "steps_run": final.get("steps_run", 0),
        "replans": final.get("replans", 0),
        "seconds": round(seconds, 3),
        "stop_reason": final.get("stop_reason", ""),
        "history": final.get("history", []),
        "waves": final.get("waves", []),
    }


def _mermaid(name: str) -> str:
    settings = Settings(provider="fake")
    if name == "executor":
        graph = build_executor(FakePlanModel(), make_toolbox(settings), tool_limit=3)
    else:
        graph = build_graph(name, settings)
    return graph.get_graph().draw_mermaid()


def main(argv: Sequence[str] | None = None, out: TextIO = sys.stdout) -> int:
    args = _parser().parse_args(argv)
    if args.mermaid:
        print(_mermaid(args.mermaid), file=out)
        return 0
    if not args.task and not args.resume:
        _parser().print_usage(sys.stderr)
        return 2
    if args.task and len(args.task) > MAX_TASK_CHARS:
        print(f"the task is longer than {MAX_TASK_CHARS} characters", file=sys.stderr)
        return 2
    if args.resume and (not args.db or not args.thread or args.variant == "all"):
        print("--resume needs --db, --thread and one --variant", file=sys.stderr)
        return 2
    try:
        settings = _settings(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    reports = []
    try:
        for variant in variants:
            if len(variants) > 1 and not args.quiet:
                print(f"\n== {variant} ==", file=out)
            reports.append(asyncio.run(_run_one(variant, args.task, settings, args, out)))
    except ModelConfigError as exc:
        print(f"model error: {exc}", file=sys.stderr)
        return 2
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: {str(exc)[:300]}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2), file=out)
        return 0
    for r in reports:
        if r["status"] == "paused":
            continue
        print(f"\n[{r['variant']}] {r['answer']}", file=out)
        print(
            f"  status={r['status']} llm_calls={r['llm_calls']} tool_calls={r['tool_calls']} "
            f"replans={r['replans']} seconds={r['seconds']}",
            file=out,
        )
    return 0
