"""Package-wide guards: no dynamic code execution, no silenced findings, no telemetry."""

from __future__ import annotations

import ast
from pathlib import Path

import plan_and_execute

PACKAGE = Path(plan_and_execute.__file__).parent
SOURCES = sorted(PACKAGE.rglob("*.py"))
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}
BANNED_MODULES = {"pickle", "marshal", "shelve", "subprocess", "yaml", "requests", "langsmith"}


def test_no_dynamic_execution_or_risky_modules() -> None:
    assert SOURCES
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in BANNED_CALLS, f"{path.name}: {node.func.id}()"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in BANNED_MODULES, (
                        f"{path.name}: {alias.name}"
                    )
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in BANNED_MODULES, (
                    f"{path.name}: {node.module}"
                )


def test_no_silenced_findings() -> None:
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for marker in ("# noqa", "# nosec", "# type: ignore"):
            assert marker not in text, f"{path.name} contains {marker!r}"


def test_bundled_notes_ship_with_the_package() -> None:
    files = sorted(p.name for p in (PACKAGE / "data" / "notes").iterdir())
    assert files == ["calendar.md", "lab.md", "offices.md", "project-atlas.md", "travel-policy.md"]
