"""The four local tools, the confined notes loader and the Toolbox."""

from __future__ import annotations

import asyncio
import os
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from plan_and_execute.notes import Notes
from plan_and_execute.tools import Toolbox
from plan_and_execute.tools import ToolError
from plan_and_execute.tools import calculate
from plan_and_execute.tools import convert_units
from plan_and_execute.tools import date_math


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("85 * 6 * 4", "2040"),
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("2,310 / 3", "770"),
        ("2^10", "1024"),
        ("6 x 7", "42"),
        ("7 // 2 + 7 % 2", "4"),
        ("-3 + +1", "-2"),
        ("sqrt(2)", "1.414214"),
        ("round(pi, 2)", "3.14"),
        ("max(1, 5, 3) - min(4, 2)", "3"),
        ("1 / 3", "0.333333"),
    ],
)
def test_calculator(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "open('/etc/passwd')",
        "(1).__class__",
        "[1, 2]",
        "lambda: 1",
        "x + 1",
        "True + 1",
        "'a' * 3",
        "9 ** 9 ** 9",
        "10 ** 100",
        "1 / 0",
        "sqrt(-1)",
        "round(1, ndigits=2)",
        "1 +",
        "",
        "1+" * 200 + "1",
        "+".join(["1"] * 40),
    ],
)
def test_calculator_refuses_anything_but_arithmetic(expression: str) -> None:
    with pytest.raises(ToolError):
        calculate(expression)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("2310 km to mi", "1435.37 mi"),
        ("Berlin to Lisbon flight distance: 2310 km to miles", "1435.37 mi"),
        ("12.5 kg in lb", "27.56 lb"),
        ("Sample freezer setpoint: -80 °C to F", "-112 °F"),
        ("212 fahrenheit to celsius", "100 °C"),
        ("0 C to kelvin", "273.15 K"),
        ("3.5 h to min", "210 min"),
        ("1 gallon to l", "3.79 l"),
        ("1,000 m to km.", "1 km"),
    ],
)
def test_unit_conversion(query: str, expected: str) -> None:
    assert convert_units(query) == expected


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("-80 °C to degrees", "unknown target unit"),
        ("12 kg to km", "cannot convert mass"),
        ("12 kg to F", "cannot convert kg to f"),
        ("12 parsecs to km", "no quantity"),
        ("12 km", "say which unit"),
        ("", "1-300"),
    ],
)
def test_unit_conversion_errors(query: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        convert_units(query)


TODAY = date(2026, 10, 24)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("2026-11-03 + 45 days", "2026-12-18 (Friday)"),
        ("Kickoff date: 2026-11-03 + Launch: 45 days after kickoff", "2026-12-18 (Friday)"),
        ("2026-12-18 - 2 weeks", "2026-12-04 (Friday)"),
        ("3 days before 2026-01-01", "2025-12-29 (Monday)"),
        ("2026-12-24", "2026-12-24 (Thursday)"),
        ("today plus 10 days", "2026-11-03 (Tuesday)"),
        ("days between 2026-01-01 and today", "296 days"),
        ("days from 2026-12-24 until 2027-01-01", "8 days"),
    ],
)
def test_date_math(query: str, expected: str) -> None:
    assert date_math(query, today=TODAY) == expected


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("next tuesday", "no date found"),
        ("2026-02-30", "not a valid date"),
        ("2026-01-01 + 99999 days", "limited"),
        ("9999-12-30 + 5 days", "out of range"),
    ],
)
def test_date_math_errors(query: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        date_math(query, today=TODAY)


def test_date_math_defaults_to_the_real_today() -> None:
    assert date_math("today").startswith(date.today().isoformat())


def test_notes_search_ranks_facts(notes: Notes) -> None:
    hits = notes.search("Berlin Lisbon flight distance", k=2)
    assert hits[0].render() == "[offices] Berlin to Lisbon flight distance: 2310 km"
    assert notes.search("zzz qqq") == []
    assert len(notes.search("Atlas", k=10)) == 6


def test_notes_loader_is_confined(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "ok.md").write_text("# Ok\n- Visible fact: 1\n", encoding="utf-8")
    (root / "skip.py").write_text("- not a note\n", encoding="utf-8")
    (root / ".hidden.md").write_text("- hidden fact\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "x.md").write_text("- git fact\n", encoding="utf-8")
    (root / "big.md").write_text("- " + "x" * 1_000_001, encoding="utf-8")
    outside = tmp_path / "secret.md"
    outside.write_text("- secret fact\n", encoding="utf-8")
    os.symlink(outside, root / "link.md")
    (root / "sub").mkdir()
    (root / "sub" / "deep.md").write_text("- Deep fact: 2\n", encoding="utf-8")
    facts = Notes.from_dir(root).facts
    assert sorted(f.render() for f in facts) == ["[ok] Visible fact: 1", "[sub/deep] Deep fact: 2"]


def test_notes_loader_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Notes.from_dir(tmp_path / "missing")
    (tmp_path / "file.md").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        Notes.from_dir(tmp_path / "file.md")
    with pytest.raises(ValueError, match="empty"):
        Notes.from_dir(tmp_path)


def test_toolbox_runs_meters_and_rejects_unknown_tools(notes: Notes) -> None:
    box = Toolbox(notes=notes, latency=0.05)

    async def three() -> list[str]:
        return list(
            await asyncio.gather(
                box.run("calculator", "1 + 1"),
                box.run("search_notes", "freezer"),
                box.run("search_notes", "zzz"),
            )
        )

    assert asyncio.run(three()) == [
        "2",
        "[lab] Sample freezer setpoint: -80 °C",
        "no matching notes",
    ]
    assert box.peak_in_flight == 3
    assert [name for name, _ in box.calls] == ["calculator", "search_notes", "search_notes"]
    with pytest.raises(ToolError, match="unknown tool"):
        asyncio.run(box.run("shell", "ls"))
    box.reset()
    assert box.calls == [] and box.peak_in_flight == 0


def test_langchain_tools_report_errors_as_error_tool_messages(notes: Notes) -> None:
    tools = {t.name: t for t in Toolbox(notes=notes).langchain_tools()}
    assert sorted(tools) == ["calculator", "convert_units", "date_math", "search_notes"]
    call = {"name": "calculator", "args": {"input": "1/0"}, "id": "c1", "type": "tool_call"}
    bad = asyncio.run(tools["calculator"].ainvoke(call))
    assert isinstance(bad, ToolMessage) and bad.status == "error"
    assert "division by zero" in bad.text
    good = tools["date_math"].invoke({**call, "name": "date_math", "args": {"input": "2026-11-03"}})
    assert good.status == "success" and good.text == "2026-11-03 (Tuesday)"
    sync_bad = tools["convert_units"].invoke({**call, "args": {"input": "1 km to kg"}})
    assert sync_bad.status == "error"
    with pytest.raises(ValueError, match="at most 300"):
        tools["calculator"].invoke({"input": "1" * 301})
