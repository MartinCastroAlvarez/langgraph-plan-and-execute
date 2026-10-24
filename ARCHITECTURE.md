# Architecture

```
plan_and_execute/
  config.py      Settings + Budget from PLAN_EXECUTE_* env vars; run_config() sizes recursion_limit
  plan.py        Step / Plan / Decision (Pydantic), extract_json, parse_plan, #E substitution
  prompts.py     system prompts ("Role: <name>." first) and data_block() for untrusted text
  tools.py       calculator (AST allowlist), convert_units, date_math, Toolbox (+ StructuredTools)
  notes.py       Markdown bullets -> facts, in-memory BM25, confined loader
  state.py       PlanState (JSON-native, reducers for parallel keys), StepResult, emit()
  common.py      shared nodes: planner (+1 repair), review (interrupt), finalize; compose_answer
  classic.py     execute (create_agent executor) <-> replan, finalize
  rewoo.py       worker loop (substitute + run, no model) -> solver
  compiler.py    schedule -> Send(task) waves -> schedule ... -> joiner -> planner | END
  graphs.py      build_graph(variant, settings, ...), make_toolbox()
  fake.py        FakePlanModel: role-aware fake BaseChatModel, scenarios, latency, scripts
  progress.py    CallCounter (model calls by role), stream_run(), format_progress()
  bench.py       three variants x three tasks: model calls, tool calls, wall clock
  cli.py         python -m plan_and_execute "<task>" [--variant ...] [--approve] [--db ...]
  data/notes/    fictional notes (one fact per bullet)
```

## One plan, three executors

```
                 ┌──────────── classic ────────────┐
planner ─ review ┼─ execute ⇄ replan ─ (finalize)  │ 1 + 3N model calls
  (JSON,         ├──────────── rewoo ──────────────┤
   validated,    │  worker ⟲ (no model) ─ solver    │ 2 model calls
   1 repair)     ├──────────── compiler ───────────┤
                 │  schedule ─Send→ task × wave ⟲   │ 2 (+2 per re-plan)
                 │  ─ joiner ─ planner | finalize   │
                 └─────────────────────────────────┘
```

- `make_planner` asks for JSON, validates it with `parse_plan` (and the budget), feeds a
  validation error back once, and otherwise ends the run with `status="failed"`.
- `make_review` is a no-op unless `approve_plan` is set. Then it calls
  `interrupt(payload, response_schema=ReviewDecision)`: LangGraph validates the resume
  value, and the node checks budget/variant rules and re-interrupts with the error.
- `finalize` never calls a model; it turns whatever results exist into an answer.

## State

`PlanState` stores only JSON-native values. The plan is `Step.model_dump()` dicts and is
re-validated by `plan_of(state)` whenever a node needs it, so any checkpointer can store
it without custom serialisers, and a resumed thread can't smuggle in an invalid plan.

| Key | Reducer | Written by |
| --- | --- | --- |
| `results` | `operator.add` | every executor/worker/task (parallel in the compiler) |
| `tool_calls`, `steps_run` | `operator.add` (deltas) | same |
| `waves` | `operator.add` | `schedule` |
| `history` | `operator.add` | planner, review, replanner, joiner |
| `steps`, `pending`, `round`, `replans`, `status`, `answer` | last write | single writers |

Results carry the `round` they belong to; the compiler's re-plans start a new round
(ids restart at 1) and `results_by_id(state, round)` reads only the current one.

## Classic executor

`build_executor` is `create_agent(model, tools, system_prompt=EXECUTOR,
middleware=[ToolCallLimitMiddleware(run_limit=k, exit_behavior="end")])`, built per step
so `k = min(max_step_tool_calls, tool calls left)`. It's invoked inside the `execute`
node, so it runs as a subgraph with the parent's callbacks and checkpointer. Tools raise
`ToolException` with `handle_tool_error=True`, so failures reach the model as
`ToolMessage(status="error")` and the step's success is read from those statuses.

## Placeholders

`#E<n>` is replaced by `evidence_value(result_n)`: the first line of the result without
a `[note-id]` prefix, capped at 300 characters, or its first number inside a calculator
expression. Tools parse leniently (first ISO date, first quantity with a known unit) so
raw evidence works as input without an extra model call.

## Compiler scheduling

`schedule` (a node) records skipped steps (dependencies failed or skipped, cascading),
picks the ready steps (all dependencies ok), truncates them to the remaining tool budget
and writes `ready` + a `waves` entry. `dispatch` (its conditional edge) turns `ready` into
`Send("task", TaskInput)` or routes to `joiner` / `finalize`. All tasks of a wave run in
one superstep, and their `results` merge through the reducer before `schedule` runs again.
