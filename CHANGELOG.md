# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-24

### Added

- One Pydantic plan format (`Step`, `Plan`) with tolerant JSON extraction, DAG
  validation (unique ids, known dependencies, no cycles), budget checks, `#E<n>`
  evidence placeholders and a single repair round for the planner.
- Classic plan-and-execute: a `create_agent` ReAct executor per step with a
  `ToolCallLimitMiddleware` cap, and a replanner that continues, finishes, inserts
  recovery steps or replaces the rest of the plan.
- ReWOO: plan once, run every tool with literal placeholder substitution and no model
  call in between, solve once; failed steps skip their dependants.
- LLMCompiler-lite: `Send` fan-out of each ready wave of the DAG, a `results` reducer,
  skip cascades, and a joiner that answers or re-plans.
- A plan-level budget (`max_steps`, `max_replans`, `max_tool_calls`,
  `max_step_tool_calls`) that always ends with a partial answer.
- Human plan review with `interrupt(..., response_schema=ReviewDecision)`: approve, edit
  or reject; resumable from `InMemorySaver` or `AsyncSqliteSaver` across processes.
- Four local tools: an AST-allowlist calculator, a unit converter, date arithmetic and
  BM25 search over bundled fictional notes.
- `FakePlanModel`, a role-aware scripted fake chat model with simulated latency.
- `python -m plan_and_execute` CLI (ollama by default, Anthropic optional, fake
  offline), `--mermaid` diagrams, durable `--db/--thread/--resume`, and
  `python -m plan_and_execute.bench`, the three-variant benchmark.
