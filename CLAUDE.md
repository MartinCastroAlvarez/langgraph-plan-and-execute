# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`langgraph-plan-and-execute` shows the planning agent architecture in current LangGraph
(plan → execute → replan with `create_agent`, ReWOO, LLMCompiler-lite with `Send`,
`interrupt()` + checkpointers, budgets) as small, correct and tested code.

## The rules

1. **Local by default.** ollama on loopback is the default; hosted providers stay
   optional adapters. No telemetry, no LangSmith upload, no web access.
2. **Model output, tool input and human edits are untrusted.** Every plan or decision
   goes through `plan.parse_plan` / `parse_decision` / `Plan(...)` validation; untrusted
   text enters prompts only through `prompts.data_block`.
3. **The graph enforces the budget, not the model.** Every loop is bounded by `Budget`;
   hitting a limit ends with `status="budget"` and an answer, never an exception.
4. **One plan format for all variants.** New variants reuse `Step`/`Plan`, the shared
   planner and the review node.
5. **No `eval`.** The calculator walks the AST against an allowlist; keep it that way.
6. **Offline tests.** `FakePlanModel` or LangChain's fake chat models only. Tests check
   control flow (calls, waves, replans, budgets, interrupts), not prose.
7. **README numbers come from real runs.** Rerun `python -m plan_and_execute.bench
   --json docs/benchmark/results.json` and paste its table. Never edit a number by hand.
8. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config. Fix the code.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`, `mypy`
(strict), `bandit` and `pip-audit` all green; `python -m examples.quickstart` runs;
README and CHANGELOG are updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets or `.env`.
