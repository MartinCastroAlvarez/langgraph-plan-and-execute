"""Four local, deterministic tools, and a :class:`Toolbox` that runs and meters them.

Every tool takes one string, so a plan step is just ``tool`` + ``input`` (ReWOO's
``Tool[input]``). Parsers are lenient about surrounding text, because ReWOO and the
compiler paste raw evidence into the next step's input, but strict about what they
compute:

* ``calculator`` walks the AST of an arithmetic expression. No ``eval``: only numbers,
  ``+ - * / // % **``, unary signs, ``pi``/``e`` and a few math functions, with caps on
  length, node count and exponent size.
* ``convert_units`` converts length, mass, volume, time and temperature.
* ``date_math`` adds or subtracts days/weeks, counts days between dates and names the
  weekday.
* ``search_notes`` searches the local notes collection.
"""

from __future__ import annotations

import ast
import asyncio
import math
import operator
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from langchain_core.tools import ToolException
from pydantic import BaseModel
from pydantic import Field

from plan_and_execute.notes import Notes

__all__ = [
    "TOOL_DESCRIPTIONS",
    "ToolError",
    "Toolbox",
    "calculate",
    "convert_units",
    "date_math",
]

MAX_INPUT_CHARS = 300


class ToolError(ValueError):
    """A tool could not handle its input. The message is safe to show to the model."""


# --------------------------------------------------------------------------- calculator
_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_MAX_NODES = 64
_MAX_EXPONENT = 64
_MAX_MAGNITUDE = 1e18
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _number(value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ToolError("only numbers are allowed")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("the result is not a finite number")
    if abs(value) > _MAX_MAGNITUDE:
        raise ToolError("the result is too large")
    return value


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return _number(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _number(_UNARY[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > _MAX_EXPONENT or abs(left) > 1e6):
            raise ToolError("exponent or base too large")
        try:
            return _number(_BINARY[type(node.op)](left, right))
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
        except OverflowError:
            raise ToolError("the result is too large") from None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
        and 1 <= len(node.args) <= 4
    ):
        args = [_evaluate(a) for a in node.args]
        try:
            return _number(_FUNCTIONS[node.func.id](*args))
        except (ValueError, TypeError):
            raise ToolError(f"invalid arguments for {node.func.id}()") from None
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def _format(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}".rstrip("0").rstrip(".")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression safely. ``^`` means power, ``x`` and ``×`` multiply."""
    text = expression.strip()
    if not text or len(text) > MAX_INPUT_CHARS:
        raise ToolError(f"the expression must be 1-{MAX_INPUT_CHARS} characters")
    text = _THOUSANDS.sub("", text).replace("×", "*").replace("^", "**")
    text = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", text)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ToolError("not an arithmetic expression") from None
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ToolError("the expression is too long")
    return _format(_evaluate(tree))


# --------------------------------------------------------------------------- units
_FACTORS: dict[str, tuple[str, float]] = {
    # length -> metres
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    "mi": ("length", 1609.344),
    "ft": ("length", 0.3048),
    "in": ("length", 0.0254),
    "yd": ("length", 0.9144),
    "nmi": ("length", 1852.0),
    # mass -> kilograms
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "mg": ("mass", 1e-6),
    "t": ("mass", 1000.0),
    "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    # volume -> litres
    "l": ("volume", 1.0),
    "ml": ("volume", 0.001),
    "gal": ("volume", 3.785411784),
    "qt": ("volume", 0.946352946),
    "cup": ("volume", 0.2365882365),
    # time -> seconds
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),
    "day": ("time", 86400.0),
    "week": ("time", 604800.0),
}
_TEMPERATURE = {"c", "f", "k"}
_ALIASES = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "centimeters": "cm",
    "millimeters": "mm",
    "mile": "mi",
    "miles": "mi",
    "foot": "ft",
    "feet": "ft",
    "inch": "in",
    "inches": "in",
    "yard": "yd",
    "yards": "yd",
    "kilogram": "kg",
    "kilograms": "kg",
    "kgs": "kg",
    "gram": "g",
    "grams": "g",
    "tonne": "t",
    "tonnes": "t",
    "pound": "lb",
    "pounds": "lb",
    "lbs": "lb",
    "ounce": "oz",
    "ounces": "oz",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "gallon": "gal",
    "gallons": "gal",
    "cups": "cup",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "mins": "min",
    "hour": "h",
    "hours": "h",
    "hr": "h",
    "days": "day",
    "d": "day",
    "weeks": "week",
    "celsius": "c",
    "fahrenheit": "f",
    "kelvin": "k",
}
_QUANTITY = re.compile(r"(?P<value>-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<unit>°?[A-Za-z]+)")
_TARGET = re.compile(r"\b(?:to|in|into)\s+(?P<unit>°?[A-Za-z]+)\s*[.?!]?\s*$", re.I)


def _unit(token: str) -> str | None:
    key = token.lower().lstrip("°")
    key = _ALIASES.get(key, key)
    return key if key in _FACTORS or key in _TEMPERATURE else None


def _to_kelvin(value: float, unit: str) -> float:
    return {"c": value + 273.15, "f": (value - 32) * 5 / 9 + 273.15, "k": value}[unit]


def _from_kelvin(value: float, unit: str) -> float:
    return {"c": value - 273.15, "f": (value - 273.15) * 9 / 5 + 32, "k": value}[unit]


def convert_units(query: str) -> str:
    """``"2310 km to mi"`` -> ``"1435.35 mi"``. Text around the quantity is ignored."""
    text = query.strip()
    if not text or len(text) > MAX_INPUT_CHARS:
        raise ToolError(f"the query must be 1-{MAX_INPUT_CHARS} characters")
    target = _TARGET.search(text)
    if target is None:
        raise ToolError("say which unit to convert to, e.g. '12 km to mi'")
    dst = _unit(target.group("unit"))
    if dst is None:
        raise ToolError(f"unknown target unit {target.group('unit')!r}")
    for match in _QUANTITY.finditer(text[: target.start()]):
        src = _unit(match.group("unit"))
        if src is None:
            continue
        value = float(match.group("value").replace(",", ""))
        if src in _TEMPERATURE or dst in _TEMPERATURE:
            if not (src in _TEMPERATURE and dst in _TEMPERATURE):
                raise ToolError(f"cannot convert {src} to {dst}")
            result = _from_kelvin(_to_kelvin(value, src), dst)
            label = dst.upper()
            return f"{round(result, 2):g} °{label}" if dst != "k" else f"{round(result, 2):g} K"
        (kind_src, f_src), (kind_dst, f_dst) = _FACTORS[src], _FACTORS[dst]
        if kind_src != kind_dst:
            raise ToolError(f"cannot convert {kind_src} ({src}) to {kind_dst} ({dst})")
        return f"{_format(round(value * f_src / f_dst, 2))} {dst}"
    raise ToolError("no quantity with a known unit before the target unit")


# --------------------------------------------------------------------------- dates
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_AFTER = re.compile(r"(\d+)\s*(days?|weeks?)\s+(after|later|from|before|earlier)", re.I)
_SIGNED = re.compile(r"(\+|-|plus|minus)\s*(\d+)\s*(days?|weeks?)", re.I)
MAX_OFFSET_DAYS = 36_500


def _dates(text: str, today: date) -> list[date]:
    found: list[tuple[int, date]] = []
    for m in _ISO.finditer(text):
        try:
            found.append((m.start(), date(int(m[1]), int(m[2]), int(m[3]))))
        except ValueError:
            raise ToolError(f"{m.group(0)} is not a valid date") from None
    for m in re.finditer(r"\btoday\b", text, re.I):
        found.append((m.start(), today))
    return [d for _, d in sorted(found, key=lambda pair: pair[0])]


def _offset(text: str) -> int | None:
    match = _AFTER.search(text)
    if match:
        n, unit, direction = int(match[1]), match[2].lower(), match[3].lower()
        sign = -1 if direction in {"before", "earlier"} else 1
    else:
        match = _SIGNED.search(text)
        if match is None:
            return None
        n, unit = int(match[2]), match[3].lower()
        sign = -1 if match[1].lower() in {"-", "minus"} else 1
    days = n * (7 if unit.startswith("week") else 1)
    if days > MAX_OFFSET_DAYS:
        raise ToolError(f"offsets are limited to {MAX_OFFSET_DAYS} days")
    return sign * days


def _show(d: date) -> str:
    return f"{d.isoformat()} ({d.strftime('%A')})"


def date_math(query: str, *, today: date | None = None) -> str:
    """``"2026-11-03 + 45 days"``, ``"days between 2026-01-01 and today"``, ``"2026-12-24"``."""
    text = query.strip()
    if not text or len(text) > MAX_INPUT_CHARS:
        raise ToolError(f"the query must be 1-{MAX_INPUT_CHARS} characters")
    now = today or date.today()
    dates = _dates(text, now)
    if not dates:
        raise ToolError("no date found; use YYYY-MM-DD or 'today'")
    if len(dates) >= 2 and re.search(r"\b(between|from|until|since)\b", text, re.I):
        return f"{abs((dates[1] - dates[0]).days)} days"
    offset = _offset(text)
    if offset is None:
        return _show(dates[0])
    try:
        return _show(dates[0] + timedelta(days=offset))
    except OverflowError:
        raise ToolError("the resulting date is out of range") from None


# --------------------------------------------------------------------------- toolbox
TOOL_DESCRIPTIONS: dict[str, str] = {
    "calculator": "Evaluate an arithmetic expression, e.g. '85 * 6 * 4' or 'sqrt(2) / 3'.",
    "search_notes": (
        "Search the local notes for facts. Input: a short query. Returns up to three "
        "'[note-id] fact' lines, best first."
    ),
    "convert_units": (
        "Convert a quantity: '<number> <unit> to <unit>' for length, mass, volume, time "
        "or temperature (C/F/K), e.g. '2310 km to mi'."
    ),
    "date_math": (
        "Date arithmetic: '2026-11-03 + 45 days', '45 days after 2026-11-03', "
        "'days between 2026-01-01 and today', or a date to get its weekday."
    ),
}


class ToolInput(BaseModel):
    input: str = Field(min_length=1, max_length=MAX_INPUT_CHARS, description="The tool input.")


@dataclass
class Toolbox:
    """Runs tools by name, with simulated latency, a call log and a concurrency gauge."""

    notes: Notes
    today: date | None = None
    latency: float = 0.0
    calls: list[tuple[str, str]] = field(default_factory=list)
    peak_in_flight: int = 0
    _in_flight: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run_sync(self, name: str, text: str) -> str:
        """Run a tool; raises :class:`ToolError` for bad input or an unknown tool."""
        text = text.strip()[:MAX_INPUT_CHARS]
        if name == "calculator":
            return calculate(text)
        if name == "convert_units":
            return convert_units(text)
        if name == "date_math":
            return date_math(text, today=self.today)
        if name == "search_notes":
            hits = self.notes.search(text, k=3)
            if not hits:
                return "no matching notes"
            return "\n".join(h.render() for h in hits)
        raise ToolError(f"unknown tool {name!r}")

    async def run(self, name: str, text: str) -> str:
        with self._lock:
            self.calls.append((name, text))
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self.latency:
                await asyncio.sleep(self.latency)
            return self.run_sync(name, text)
        finally:
            with self._lock:
                self._in_flight -= 1

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()
            self.peak_in_flight = self._in_flight = 0

    def langchain_tools(self) -> list[BaseTool]:
        """The same tools as LangChain ``StructuredTool``s for the ReAct executor."""
        return [self._tool(name) for name in TOOL_DESCRIPTIONS]

    def _tool(self, name: str) -> BaseTool:
        async def coroutine(input: str) -> str:
            try:
                return await self.run(name, input)
            except ToolError as exc:
                raise ToolException(f"error: {exc}") from None

        def func(input: str) -> str:
            try:
                return self.run_sync(name, input)
            except ToolError as exc:
                raise ToolException(f"error: {exc}") from None

        return StructuredTool.from_function(
            func=func,
            coroutine=coroutine,
            name=name,
            description=TOOL_DESCRIPTIONS[name],
            args_schema=ToolInput,
            handle_tool_error=True,
        )
