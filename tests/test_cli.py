"""The CLI: offline runs, comparison, mermaid, interactive and durable plan review."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from plan_and_execute.cli import main
from tests.conftest import FREEZER
from tests.conftest import LAUNCH
from tests.conftest import step


def _run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


@pytest.mark.parametrize("variant", ["classic", "rewoo", "compiler", "executor"])
def test_mermaid(variant: str) -> None:
    code, text = _run("--mermaid", variant)
    assert code == 0 and text.startswith("---") and "graph TD;" in text


def test_compare_all_variants_as_json() -> None:
    code, text = _run("--provider", "fake", "--variant", "all", "--json", "--quiet", LAUNCH)
    assert code == 0
    reports = json.loads(text)
    assert [r["variant"] for r in reports] == ["classic", "rewoo", "compiler"]
    assert [r["llm_calls"] for r in reports] == [16, 2, 2]
    assert reports[2]["waves"] == [[1, 2, 4], [3, 5]]


def test_progress_and_summary_lines() -> None:
    code, text = _run(
        "--provider", "fake", "--variant", "compiler", "--latency", "0", "--tool-latency", "0",
        "--today", "2026-10-24", "--max-steps", "6", "--max-replans", "1", "--max-tool-calls", "9",
        FREEZER,
    )  # fmt: skip
    assert code == 0
    assert "wave" in text and "replan" in text
    assert "[compiler] Sample freezer setpoint: -80 °C, which is -112 °F." in text
    assert "llm_calls=4 tool_calls=4 replans=1" in text


def test_interactive_review(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["edit:/does/not/exist.json", "reject"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    code, text = _run("--provider", "fake", "--approve", FREEZER)
    assert code == 0
    assert text.count("Plan awaiting review:") == 2
    assert "status=rejected" in text


def test_durable_review_across_two_invocations(tmp_path: Path) -> None:
    db = str(tmp_path / "runs.sqlite")
    common = ("--provider", "fake", "--variant", "rewoo", "--db", db, "--thread", "t1")
    code, text = _run(*common, "--approve", FREEZER)
    assert code == 0 and "Paused for review (thread t1)" in text
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"steps": [step(1, "search_notes", "freezer"), step(2, "convert_units", "#E1 to F")]}
        ),
        encoding="utf-8",
    )
    code, text = _run(*common, "--resume", f"edit:{plan}")
    assert code == 0 and "which is -112 °F." in text


def test_resume_with_a_malformed_plan_reports_an_error(tmp_path: Path) -> None:
    db = str(tmp_path / "runs.sqlite")
    common = ("--provider", "fake", "--db", db, "--thread", "t2")
    assert _run(*common, "--approve", FREEZER)[0] == 0
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([step(1, None, None, [1])]), encoding="utf-8")
    assert _run(*common, "--resume", f"edit:{plan}")[0] == 1
    assert _run(*common, "--resume", "approve", "--quiet")[0] == 0


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["x" * 5000],
        ["--resume", "approve"],
        ["--db", "x.sqlite", "--thread", "t", "--variant", "all", "--resume", "approve"],
        ["--max-steps", "0", "task"],
    ],
)
def test_usage_errors(argv: list[str]) -> None:
    assert main(argv, out=io.StringIO()) == 2


def test_resume_spec_errors(tmp_path: Path) -> None:
    db = str(tmp_path / "x.sqlite")
    code, _ = _run("--provider", "fake", "--db", db, "--thread", "t", "--resume", "maybe")
    assert code == 1


def test_remote_ollama_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://203.0.113.9:11434")
    assert main(["--provider", "ollama", "task"], out=io.StringIO()) == 2
