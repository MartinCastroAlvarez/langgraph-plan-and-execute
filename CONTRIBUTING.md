# Contributing to langgraph-plan-and-execute

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install                       # add -E anthropic for the optional adapter
poetry run pytest
poetry run ruff check plan_and_execute tests examples && poetry run ruff format --check plan_and_execute tests examples
poetry run mypy plan_and_execute
poetry run bandit -c pyproject.toml -q -r plan_and_execute
poetry run pip-audit
poetry run python -m examples.quickstart
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests that call a real model or touch the network.
- Hosted providers, web tools or tracing turned on by default.
- Tools that execute code, touch the filesystem outside the notes root, or reach the
  network.
- A loop that the budget doesn't bound.
- Hand-edited benchmark numbers.

## What it does accept

- New planning variants (e.g. eager LLMCompiler dispatch, streaming planners,
  tree-of-plans), reusing the shared plan format, with tests for control flow, budgets
  and failure handling.
- Better local tools, with hostile-input tests.

## Updating the benchmark table

```bash
poetry run python -m plan_and_execute.bench --json docs/benchmark/results.json
```

Paste the printed table into the README together with the machine, versions and date.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
