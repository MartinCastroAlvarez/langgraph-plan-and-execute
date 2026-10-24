"""A local notes collection: one searchable fact per bullet, ranked with BM25.

The loader is confined to its root folder: it reads only regular ``.md``/``.txt`` files,
skips symlinks and hidden entries, checks that every file resolves inside the root, and
caps file size and count. Search runs over an in-memory index; the tool that wraps it
never sees a path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

__all__ = ["Fact", "Notes", "bundled_notes_dir"]

MAX_FILE_BYTES = 1_000_000
MAX_FILES = 500
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a", "an", "and", "are", "at", "be", "by", "do", "does", "for", "from", "how", "in",
        "is", "it", "many", "much", "of", "on", "or", "the", "to", "what", "when", "where",
        "which", "who", "why", "with",
    }
)  # fmt: skip


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def bundled_notes_dir() -> Path:
    return Path(str(resources.files("plan_and_execute") / "data" / "notes"))


@dataclass(frozen=True, slots=True)
class Fact:
    note: str
    title: str
    text: str

    def render(self) -> str:
        return f"[{self.note}] {self.text}"


class Notes:
    """BM25 over facts. ``title`` words are indexed with every fact of the note."""

    def __init__(self, facts: list[Fact], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not facts:
            raise ValueError("the notes collection is empty")
        self.facts = facts
        self._k1, self._b = k1, b
        self._docs = [Counter(_tokens(f"{f.title} {f.text}")) for f in facts]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lengths) / len(self._lengths)
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(facts)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @classmethod
    def from_dir(cls, root: str | Path | None = None) -> Notes:
        base = Path(root) if root is not None else bundled_notes_dir()
        base = base.resolve(strict=True)
        if not base.is_dir():
            raise ValueError(f"{base} is not a directory")
        facts: list[Fact] = []
        files = 0
        for path in sorted(base.rglob("*")):
            rel = path.relative_to(base)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if path.is_symlink() or not path.is_file() or path.suffix not in {".md", ".txt"}:
                continue
            if not path.resolve().is_relative_to(base):
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            files += 1
            if files > MAX_FILES:
                break
            facts.extend(_parse(rel.with_suffix("").as_posix(), path.read_text("utf-8")))
        return cls(facts)

    def search(self, query: str, k: int = 3) -> list[Fact]:
        terms = _tokens(query)
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            score = 0.0
            norm = self._k1 * (1 - self._b + self._b * self._lengths[i] / self._avg)
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self._idf[term] * tf * (self._k1 + 1) / (tf + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self.facts[i] for _, i in scored[:k]]


def _parse(note: str, text: str) -> list[Fact]:
    title = note
    facts = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("# ") and title == note:
            title = line[2:].strip()
        elif line.startswith(("- ", "* ")) and len(line) > 2:
            facts.append(Fact(note=note, title=title, text=line[2:].strip()[:500]))
    return facts
