# Security

langgraph-plan-and-execute runs LLM agents that plan, call local tools and read a folder
of notes. People install it and point it at their own notes and models. Design goals:
nothing leaves the machine unless you opt in, text in the notes, the task or a model
reply can't take over the agent or the host, and no plan can run without bound.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/langgraph-plan-and-execute/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The notes folder, the tasks you run, the local machine and filesystem, and
your model endpoint (and its key, if you use a hosted one).

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| A note author | the text of a file in the notes folder | give the agent instructions, make the loader read files outside the folder |
| Whoever writes the task | the task text | escape the prompt structure, run code, exhaust the host |
| The model | plans, decisions, tool inputs, answers | run code, read files, loop forever, exceed the budget |
| A plan reviewer | the resume value of the interrupt | inject an invalid or over-budget plan |
| Whoever controls `OLLAMA_HOST` | where prompts are sent | receive your notes without your explicit opt-in |

| Threat | Control |
| --- | --- |
| Prompt injection (notes, task, tool results, model output) | Every system prompt says `<data>` content is untrusted and never instructions. Untrusted text is wrapped in `<data name=...>` after escaping `& < >`, so it can't close the element (tested). Escaping reduces injection but can't remove it: treat answers about untrusted notes as untrusted too. |
| Code execution through the calculator | No `eval`/`exec`: the expression is parsed with `ast.parse(mode="eval")` and walked against an allowlist (numbers, `+ - * / // % **`, unary signs, `pi`, `e`, nine math functions without keywords). Names, attributes, subscripts, lambdas, strings and comprehensions are refused. Input ≤ 300 chars, ≤ 64 AST nodes, exponent ≤ 64, base ≤ 1e6, results ≤ 1e18 and finite. Tested with `__import__`, `open`, `(1).__class__` and power towers. |
| Tool abuse by the model | Four tools only, each taking one string ≤ 300 chars (pydantic-validated). None touches the network or the filesystem; `search_notes` searches an in-memory index. Date offsets are capped at 100 years. |
| Path traversal / symlink escape in the notes loader | The root is resolved first. Only regular `.md`/`.txt` files are read; symlinks and hidden entries (`.git`, `.env`) are skipped; every file must resolve inside the root; files over 1 MB and more than 500 files are ignored. Tested with a symlink pointing outside. |
| Malicious or broken plans | Every plan (planner, replanner, joiner, human edit) is validated: known tool names only, ≤ 20 steps and ≤ `max_steps`, unique ids, known dependencies, no cycles, bounded string lengths. Invalid model plans get one repair round, then the run fails before any tool runs. Invalid human edits are refused (`ValidationError`) or sent back with the error, at most three times. |
| Runaway loops | `max_steps`, `max_replans`, `max_tool_calls` and `max_step_tool_calls` are enforced by the graph, not the model; hitting one ends the run with a partial answer. `recursion_limit` is derived from the budget as a backstop. The CLI refuses tasks over 4,000 characters and plan files over 100 KB. |
| Notes leaving the machine | `OLLAMA_HOST` must be loopback unless `PLAN_EXECUTE_ALLOW_REMOTE_OLLAMA=1`. Hosted providers are opt-in and need your own key. LangSmith tracing stays off unless you set `LANGSMITH_TRACING` yourself. |
| Checkpoint contents | Checkpoints (`--db`) contain the task, the plan and tool results in plain SQLite. Keep the file where your notes are allowed to be. State holds JSON-native values only, and plans are re-validated when read back. |
| Secrets | No key is read except `ANTHROPIC_API_KEY`, and only by the optional adapter. Keys never enter prompts, state, stream events, checkpoints or logs. `.env` is git-ignored. |
| Unsafe code in the package | An AST test bans `eval`, `exec`, `compile`, `__import__`, `pickle`, `marshal`, `shelve`, `subprocess`, `yaml` and `requests`, and any `noqa`/`nosec`/`type: ignore`. |

## 2. Dependencies

CI runs `bandit` on the package and `pip-audit` on the locked environment. Dependabot
alerts, security updates and secret scanning are enabled on the repository.

## 3. Running it safely

- Keep ollama on `127.0.0.1`. If you point `OLLAMA_HOST` elsewhere, use TLS and trust
  that host with your notes.
- Use `--approve` when the notes or tasks come from people you don't trust: you see the
  plan before any tool runs.
- Treat answers as drafts and check them against the step results, which are kept in
  the final state.
