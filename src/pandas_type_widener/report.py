"""Aggregate cell results into a per-argument support summary, as text and JSON.

Results are grouped by target, argument, axis and column. Within a group, a dtype's
verdict is its *worst* verdict over every call site that tried it. A dtype is
"accepted" only if every probed call accepted it: it is the safe set, not the
seen-working-once set.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dtypes import collapse

__all__ = ["NOTE", "Summary", "render", "summarise", "write_json"]

NOTE = (
    "'accepts' means the call returned without raising, with nothing degraded; it does not "
    "mean the result is correct. The test oracle re-runs the test's own assertions, "
    "which can fail on values that are only wrong for the dtype the test was written for."
)

_CALL_RANK = {"accepted": 0, "degraded": 1, "rejected": 2}
_TEST_RANK = {"passed": 0, "failed": 1, "call-raised": 2}


@dataclass
class Summary:
    """The support summary for one target argument along one axis.

    Attributes
    ----------
    target : str
        Target display name.
    slot : str
        Argument (parameter) name.
    axis : str
        ``column``, ``series``, ``index`` or ``joint``.
    column : str or None
        Column label, for ``column``.
    """

    target: str
    slot: str
    axis: str
    column: str | None
    sites: set[tuple[str, int]] = field(default_factory=set)
    tested: list[str] = field(default_factory=list)
    universe: dict[str, str] = field(default_factory=dict)
    call: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing: dict[str, dict[str, Any]] = field(default_factory=dict)
    test: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Return how the argument and axis read in a report.

        Returns
        -------
        str
            E.g. ``df['a']``, ``df.index``, ``s``, ``df[*]``.
        """
        if self.axis == "column":
            return f"{self.slot}[{self.column!r}]"
        if self.axis == "index":
            return f"{self.slot}.index"
        if self.axis == "joint":
            return f"{self.slot}[*]"
        return self.slot

    def names(self, oracle: str, status: str) -> list[str]:
        """Return the dtypes with a given verdict under one oracle, in grid order.

        Parameters
        ----------
        oracle : {"call", "missing", "test"}
            Which oracle's verdicts.
        status : str
            The verdict status.

        Returns
        -------
        list of str
            Matching dtype names.
        """
        verdicts = getattr(self, oracle)
        return [
            name
            for name in self.universe
            if verdicts.get(name, {}).get("status") == status
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dict, with collapsed family summaries.

        Returns
        -------
        dict
            The summary.
        """
        out = asdict(self)
        out["sites"] = len(self.sites)
        out["label"] = self.label
        out["accepts"] = collapse(self.names("call", "accepted"), self.universe)
        out["degrades"] = collapse(self.names("call", "degraded"), self.universe)
        out["rejects"] = collapse(self.names("call", "rejected"), self.universe)
        out["test_passes"] = collapse(self.names("test", "passed"), self.universe)
        return out


def _worse(
    rank: dict[str, int], old: dict[str, Any] | None, new: dict[str, Any] | None
) -> bool:
    if new is None or new["status"] not in rank:
        return False
    return old is None or rank[new["status"]] > rank[old["status"]]


def summarise(cells: Iterable[dict[str, Any]]) -> list[Summary]:
    """Group cell results (as dicts) into summaries.

    Parameters
    ----------
    cells : iterable of dict
        :meth:`CellResult.to_dict` outputs.

    Returns
    -------
    list of Summary
        One per target, argument, axis and column, in first-seen order.
    """
    groups: dict[tuple[Any, ...], Summary] = {}
    for cell in cells:
        key = (cell["target"], cell["slot"], cell["axis"], cell["column"])
        summary = groups.get(key)
        if summary is None:
            summary = groups[key] = Summary(*key)
        summary.sites.add((cell["test"], cell["ordinal"]))
        if cell["tested"] not in summary.tested:
            summary.tested.append(cell["tested"])
        name = cell["dtype"]
        summary.universe.setdefault(name, cell["family"])
        for oracle, attr, rank in (
            ("call", "call", _CALL_RANK),
            ("call_missing", "missing", _CALL_RANK),
            ("test_oracle", "test", _TEST_RANK),
        ):
            verdicts = getattr(summary, attr)
            if _worse(rank, verdicts.get(name), cell.get(oracle)):
                verdicts[name] = cell[oracle]
    return list(groups.values())


def _exceptions(summary: Summary, oracle: str, names: list[str]) -> str:
    verdicts = getattr(summary, oracle)
    counts = Counter(
        verdicts[n]["detail"].split(":", 1)[0] for n in names if verdicts[n]["detail"]
    )
    if not counts:
        return ""
    return (
        " ("
        + ", ".join(f"{exc} x{n}" if n > 1 else exc for exc, n in counts.most_common())
        + ")"
    )


def _line(head: str, verb: str, items: list[str], suffix: str = "") -> str:
    return f"  {head:<8} {verb:<9}{', '.join(items)}{suffix}"


def render(summaries: list[Summary]) -> list[str]:
    """Render summaries as terminal lines.

    Parameters
    ----------
    summaries : list of Summary
        What :func:`summarise` returned.

    Returns
    -------
    list of str
        Lines, without trailing newlines.
    """
    lines: list[str] = []
    for summary in summaries:
        sites = len(summary.sites)
        lines.append(
            f"{summary.target}  {summary.label}  tested {', '.join(summary.tested)}"
            f"  ({sites} call site{'s' if sites != 1 else ''})"
        )
        u = summary.universe
        accepted = summary.names("call", "accepted")
        degraded = summary.names("call", "degraded")
        rejected = summary.names("call", "rejected")
        passed = summary.names("test", "passed")
        broken = summary.names("missing", "rejected")
        # Only news belongs on the missing-values line: a dtype that already degrades
        # without missing values degrading with them too says nothing new.
        soft = [n for n in summary.names("missing", "degraded") if n in accepted]
        everything = list(u)
        if (
            accepted == everything
            and not broken
            and not soft
            and (not summary.test or passed == everything)
        ):
            # Typically a column the target never reads. One line is enough.
            lines[-1] += "  ->  accepts any"
            continue
        if summary.call:
            lines.append(_line("call", "accepts", collapse(accepted, u) or ["nothing"]))
            if degraded:
                reasons = sorted({summary.call[n]["detail"] for n in degraded})
                why = f" ({reasons[0]}{'; ...' if len(reasons) > 1 else ''})"
                lines.append(_line("", "degrades", collapse(degraded, u), why))
            if rejected:
                lines.append(
                    _line(
                        "",
                        "rejects",
                        collapse(rejected, u),
                        _exceptions(summary, "call", rejected),
                    )
                )
        if broken:
            lines.append(
                _line(
                    "missing",
                    "breaks",
                    collapse(broken, u),
                    _exceptions(summary, "missing", broken),
                )
            )
        if soft:
            lines.append(
                _line("missing" if not broken else "", "degrades", collapse(soft, u))
            )
        if summary.test:
            failed = summary.names("test", "failed")
            raised = summary.names("test", "call-raised")
            lines.append(_line("test", "passes", collapse(passed, u) or ["nothing"]))
            if failed:
                lines.append(
                    _line(
                        "",
                        "fails",
                        collapse(failed, u),
                        _exceptions(summary, "test", failed),
                    )
                )
            if raised:
                lines.append(
                    _line(
                        "",
                        "raises",
                        collapse(raised, u),
                        _exceptions(summary, "test", raised),
                    )
                )
    if summaries:
        lines += ["", f"note: {NOTE}"]
    return lines


def write_json(
    path: str | Path, cells: list[dict[str, Any]], errors: list[str]
) -> None:
    """Write every cell result and the summaries to a JSON file.

    Parameters
    ----------
    path : str or Path
        Where to write.
    cells : list of dict
        :meth:`CellResult.to_dict` outputs.
    errors : list of str
        Internal errors met while probing.
    """
    payload = {
        "version": 1,
        "note": NOTE,
        "summaries": [s.to_dict() for s in summarise(cells)],
        "cells": cells,
        "errors": errors,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=_jsonable))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return sorted(value, key=str)
    return str(value)
