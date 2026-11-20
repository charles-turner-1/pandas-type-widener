"""Replay recorded calls with one argument's dtypes changed, and judge each replay.

For each :class:`~pandas_type_widener.runtime.CallSite`, every DataFrame or Series
argument is varied along these **axes**:

- each column of a frame, one at a time, keeping the other columns and the index as
  they were recorded;
- the values of a Series;
- the index;
- optionally (*joint*), every column of a frame at once, cast to one representative
  dtype per family.

Each axis is crossed with the dtype grid, giving **cells**. Each cell is judged by up to
three verdicts:

``call``
    The target is called directly with the changed argument. *accepted*: it returned.
    *degraded*: it returned, but warned, came back all-missing, or lost a numeric result.
    *rejected*: it raised.
``call_missing``
    The same, with missing values injected into the column. Hypothesis chooses where.
``test``
    The whole test is re-run with the changed argument swapped into the same call
    (supplied by the plugin through ``rerun``). *passed*, *failed* (an assertion or a
    later step failed), *call-raised* (the target raised), or *unmatched* (the re-run
    didn't make the same call).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from hypothesis import HealthCheck, Phase, Verbosity, given, settings
from hypothesis import strategies as st
from hypothesis.errors import HypothesisException

from .dtypes import (
    DtypeSpec,
    IndexSpec,
    describe_index,
    describe_series,
    is_missing,
)
from .runtime import CallSite, Injection, ResultSummary, Slot

__all__ = ["Axis", "CellResult", "Prober", "Verdict"]

# Distinct values drawn per column. Longer columns repeat the drawn values: the dtype is
# what's under test, not long-range structure.
_MAX_DRAWN = 20
_MAX_MISSING = 5
_JOINT_REPRESENTATIVES = (
    "bool",
    "int64",
    "float64",
    "str",
    "object[str]",
    "category[str]",
    "datetime64[ns]",
)


@dataclass
class Verdict:
    """The outcome of one cell under one oracle.

    Attributes
    ----------
    status : str
        ``accepted``/``degraded``/``rejected`` for the call oracles;
        ``passed``/``failed``/``call-raised``/``unmatched``/``skipped`` for the test
        oracle.
    detail : str
        The exception or reason, if any.
    example : str, optional
        A short rendering of the input that produced this verdict.
    """

    status: str
    detail: str = ""
    example: str | None = None


@dataclass
class CellResult:
    """Everything known about one cell: one call site, one axis, one dtype.

    Attributes
    ----------
    test : str
        Test node id.
    target : str
        Target display name.
    ordinal : int
        Which outermost target call in the test.
    slot : str
        The argument's parameter name.
    axis : str
        ``column``, ``series``, ``index`` or ``joint``.
    column : str or None
        Column label for a ``column`` axis.
    tested : str
        The dtype the test itself used on this axis.
    dtype : str
        The dtype tried.
    family : str
        The tried dtype's family.
    source : str
        ``cast`` (the test's values, converted), ``drawn`` (Hypothesis values) or
        ``fixed`` (a deterministic index).
    call : Verdict, optional
        Call-oracle verdict.
    call_missing : Verdict, optional
        Call-oracle verdict with missing values.
    test_oracle : Verdict, optional
        Test-oracle verdict.
    """

    test: str
    target: str
    ordinal: int
    slot: str
    axis: str
    column: str | None
    tested: str
    dtype: str
    family: str
    source: str
    call: Verdict | None = None
    call_missing: Verdict | None = None
    test_oracle: Verdict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dict.

        Returns
        -------
        dict
            The result, with verdicts as nested dicts.
        """
        return asdict(self)


@dataclass(frozen=True)
class Axis:
    """One way of varying an argument.

    Attributes
    ----------
    kind : str
        ``column``, ``series``, ``index`` or ``joint``.
    tested : str
        What the recorded argument has on this axis.
    position : int, optional
        Column position, for ``column``.
    label : str, optional
        Column label, for ``column``.
    """

    kind: str
    tested: str
    position: int | None = None
    label: str | None = None

    def original(self, value: Any) -> pd.Series:
        """Return the recorded column (or Series) this axis varies.

        Parameters
        ----------
        value : DataFrame or Series
            The recorded argument.

        Returns
        -------
        pandas.Series
            The column.
        """
        if self.kind == "column":
            assert self.position is not None
            column: pd.Series = value.iloc[:, self.position]
            return column
        return value

    def apply(self, value: Any, replacement: Any) -> Any:
        """Return a copy of ``value`` with this axis replaced.

        Parameters
        ----------
        value : DataFrame or Series
            The argument as it is (recorded, or live in a re-run).
        replacement : Series, Index or dict
            New column (``column``/``series``), new index (``index``), or column
            position to new column (``joint``).

        Returns
        -------
        DataFrame or Series
            The changed argument.

        Raises
        ------
        ValueError
            If ``replacement`` doesn't fit ``value``.
        """
        if self.kind == "index":
            if len(replacement) != len(value):
                raise ValueError("index length changed")
            return value.set_axis(replacement, axis=0)
        if self.kind == "series":
            if len(replacement) != len(value):
                raise ValueError("series length changed")
            return pd.Series(replacement.array, index=value.index, name=value.name)
        columns = {self.position: replacement} if self.kind == "column" else replacement
        new = value.copy()
        for position, column in columns.items():
            if len(column) != len(new) or position >= new.shape[1]:
                raise ValueError("frame shape changed")
            new.isetitem(position, column.array)
        return new


def axes(value: Any, joint: bool) -> list[Axis]:
    """List the axes along which an argument is varied.

    Parameters
    ----------
    value : DataFrame or Series
        The recorded argument.
    joint : bool
        Whether to add the all-columns-at-once axis for frames.

    Returns
    -------
    list of Axis
        Columns (or the series), then the index, then the joint axis.
    """
    out: list[Axis] = []
    if isinstance(value, pd.DataFrame):
        for position, label in enumerate(value.columns):
            tested = describe_series(value.iloc[:, position])
            out.append(Axis("column", tested, position, str(label)))
    else:
        out.append(Axis("series", describe_series(value)))
    out.append(Axis("index", describe_index(value.index)))
    if joint and isinstance(value, pd.DataFrame) and value.shape[1] > 1:
        tested = "/".join(
            sorted({describe_series(value.iloc[:, i]) for i in range(value.shape[1])})
        )
        out.append(Axis("joint", tested))
    return out


class _Rejected(Exception):
    """Raised inside a Hypothesis run to report a rejecting example."""


class Prober:
    """Runs every cell of every recorded call site.

    Parameters
    ----------
    grid : list of DtypeSpec
        Column dtypes to try.
    index_grid : list of IndexSpec
        Index types to try.
    examples : int
        Hypothesis examples per randomised check.
    joint : bool
        Whether to vary all columns of a frame together too.
    """

    def __init__(
        self,
        grid: list[DtypeSpec],
        index_grid: list[IndexSpec],
        examples: int,
        joint: bool = False,
    ) -> None:
        self.grid = grid
        self.index_grid = index_grid
        self.examples = examples
        self.joint = joint

    def probe(
        self,
        site: CallSite,
        rerun: Callable[[Injection], Verdict] | None = None,
    ) -> Iterator[CellResult]:
        """Yield a result for every cell of one call site.

        Parameters
        ----------
        site : CallSite
            The recorded call.
        rerun : callable, optional
            Runs the test again with an :class:`Injection` and returns its verdict. If
            omitted, the test oracle is skipped.

        Yields
        ------
        CellResult
            One per axis and dtype.
        """
        for slot in site.slots:
            recorded = slot.get(site.args, site.kwargs)
            _, exc, baseline_warnings = self._call(site, slot, recorded.copy(deep=True))
            if exc is not None:
                # The recorded call doesn't replay (it depends on state that's gone, say).
                # Nothing measured against it would mean anything.
                continue
            for axis in axes(recorded, self.joint):
                yield from self._probe_axis(
                    site, slot, recorded, axis, baseline_warnings, rerun
                )

    def _probe_axis(
        self,
        site: CallSite,
        slot: Slot,
        recorded: Any,
        axis: Axis,
        baseline_warnings: set[str],
        rerun: Callable[[Injection], Verdict] | None,
    ) -> Iterator[CellResult]:
        def cell(name: str, family: str, source: str) -> CellResult:
            return CellResult(
                test=site.nodeid,
                target=site.target,
                ordinal=site.ordinal,
                slot=slot.name,
                axis=axis.kind,
                column=axis.label,
                tested=axis.tested,
                dtype=name,
                family=family,
                source=source,
            )

        def evaluate(replacement: Any) -> Verdict:
            return self._evaluate(
                site, slot, axis.apply(recorded, replacement), baseline_warnings
            )

        def evaluate_missing(replacement: Any) -> Verdict:
            value = axis.apply(recorded, replacement)
            return self._evaluate(
                site, slot, value, baseline_warnings, with_missing=True
            )

        def test_oracle(replacement: Any) -> Verdict | None:
            if rerun is None:
                return None
            plan = Injection(
                site.ordinal,
                site.target,
                slot,
                lambda live: axis.apply(live, replacement),
            )
            return rerun(plan)

        if axis.kind == "index":
            for ispec in self.index_grid:
                result = cell(ispec.name, ispec.family, "fixed")
                replacement = ispec.make(len(recorded))
                result.call = evaluate(replacement)
                result.test_oracle = test_oracle(replacement)
                yield result
            return

        if axis.kind == "joint":
            by_name = {spec.name: spec for spec in self.grid}
            for name in _JOINT_REPRESENTATIVES:
                spec = by_name.get(name)
                if spec is None:
                    continue
                columns = {}
                for position in range(recorded.shape[1]):
                    cast = spec.cast(recorded.iloc[:, position])
                    if cast is None:
                        break
                    columns[position] = cast
                else:
                    result = cell(spec.name, spec.family, "cast")
                    result.call = evaluate(columns)
                    result.test_oracle = test_oracle(columns)
                    yield result
            return

        original = axis.original(recorded)
        for spec in self.grid:
            cast = spec.cast(original)
            result = cell(
                spec.name, spec.family, "cast" if cast is not None else "drawn"
            )
            if cast is not None:
                result.call = evaluate(cast)
                representative = cast
            else:
                result.call, representative = self._drawn(spec, original, evaluate)
            if result.call.status != "rejected" and spec.nullable and len(original):
                result.call_missing = self._missing(
                    spec, original, cast, evaluate_missing
                )
            if representative is not None:
                result.test_oracle = test_oracle(representative)
            yield result

    @staticmethod
    def _call(
        site: CallSite, slot: Slot, value: Any
    ) -> tuple[Any, Exception | None, set[str]]:
        """Call the target once with ``value`` in ``slot``.

        Returns the result (or None), the exception raised (or None), and the warnings
        emitted, as ``"Category: message"`` strings.
        """
        args, kwargs = site.fresh_args()
        args, kwargs = slot.replace(args, kwargs, value)
        result: Any = None
        error: Exception | None = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = site.func(*args, **kwargs)
            except Exception as exc:
                error = exc
        return result, error, {f"{w.category.__name__}: {w.message}" for w in caught}

    def _evaluate(
        self,
        site: CallSite,
        slot: Slot,
        value: Any,
        baseline_warnings: set[str],
        with_missing: bool = False,
    ) -> Verdict:
        """Call the target once with ``value`` in ``slot`` and judge the outcome.

        With ``with_missing``, an all-missing result is not held against the dtype:
        missing values in, missing values out is propagation, not breakage.
        """
        result, error, messages = self._call(site, slot, value)
        if error is not None:
            return Verdict("rejected", _describe_exc(error), _render(value))
        reasons = sorted(messages - baseline_warnings)
        reasons += _result_changes(site.result, ResultSummary.of(result), with_missing)
        if reasons:
            return Verdict("degraded", "; ".join(reasons), _render(value))
        return Verdict("accepted")

    def _hypothesis(
        self, strategy: st.SearchStrategy[Any], evaluate: Callable[[Any], Verdict]
    ) -> tuple[Verdict, Any]:
        """Search ``strategy`` for an input the target rejects.

        Returns the verdict, and an input that represents it: the shrunk rejecting
        input, or else the first one tried.
        """
        failures: list[Verdict] = []
        failing_inputs: list[Any] = []
        seen: list[Any] = []
        degraded: list[Verdict] = []

        @settings(
            max_examples=self.examples,
            deadline=None,
            derandomize=True,  # reports must not change between runs
            database=None,
            suppress_health_check=list(HealthCheck),
            phases=[Phase.generate, Phase.shrink],
            verbosity=Verbosity.quiet,
            print_blob=False,
        )
        @given(strategy)
        def check(value: Any) -> None:
            if not seen:
                seen.append(value)
            verdict = evaluate(value)
            if verdict.status == "rejected":
                failures.append(verdict)
                failing_inputs.append(value)
                raise _Rejected
            if verdict.status == "degraded" and not degraded:
                degraded.append(verdict)

        try:
            check()
        except _Rejected:
            return failures[-1], failing_inputs[-1]
        except HypothesisException as exc:
            if failures:  # e.g. flaky: report the last rejection seen
                return failures[-1], failing_inputs[-1]
            return Verdict("skipped", f"hypothesis: {type(exc).__name__}"), None
        except Exception as exc:
            # ``evaluate`` turns the target's own exceptions into verdicts, so anything
            # else came from building the input: our problem, not the library's.
            return (
                Verdict("skipped", f"couldn't build an input: {_describe_exc(exc)}"),
                None,
            )
        if degraded:
            return degraded[0], seen[0] if seen else None
        return Verdict("accepted"), seen[0] if seen else None

    def _drawn(
        self, spec: DtypeSpec, original: pd.Series, evaluate: Callable[[Any], Verdict]
    ) -> tuple[Verdict, Any]:
        """Judge ``spec`` on fresh values, for a column that won't cast to it."""
        n = len(original)
        if n == 0:
            empty = spec.build([], original.index, original.name)
            return evaluate(empty), empty
        values = st.lists(spec.elements, min_size=1, max_size=min(n, _MAX_DRAWN))
        strategy = values.map(
            lambda vs: spec.build(_tile(vs, n), original.index, original.name)
        )
        return self._hypothesis(strategy, evaluate)

    def _missing(
        self,
        spec: DtypeSpec,
        original: pd.Series,
        cast: pd.Series | None,
        evaluate: Callable[[Any], Verdict],
    ) -> Verdict:
        """Judge ``spec`` with missing values at positions Hypothesis chooses."""
        n = len(original)
        positions = st.sets(
            st.integers(0, n - 1), min_size=1, max_size=min(n, _MAX_MISSING)
        )
        if cast is not None:
            base = cast

            def with_missing(where: set[int]) -> pd.Series:
                mask = [i in where for i in range(n)]
                return base.where(~pd.Series(mask, index=base.index))

            strategy = positions.map(with_missing)
        else:
            values = st.lists(spec.elements, min_size=1, max_size=min(n, _MAX_DRAWN))

            def build(drawn: tuple[list[Any], set[int]]) -> pd.Series:
                cells = _tile(drawn[0], n)
                for i in drawn[1]:
                    cells[i] = None
                return spec.build(cells, original.index, original.name)

            strategy = st.tuples(values, positions).map(build)
        verdict, _ = self._hypothesis(strategy, evaluate)
        return verdict


def _tile(values: list[Any], n: int) -> list[Any]:
    return [values[i % len(values)] for i in range(n)]


def _describe_exc(exc: BaseException) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if len(message) > 160:
        message = message[:157] + "..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _render(value: Any) -> str:
    """Render the first few rows of an argument, for a report."""
    try:
        head = value.head(3)
        if isinstance(head, pd.Series):
            text = ", ".join("NA" if is_missing(v) else repr(v) for v in head)
            return f"[{text}{', ...' if len(value) > 3 else ''}] ({describe_series(value)})"
        return str(head.to_string(max_cols=6))
    except Exception:  # pragma: no cover - rendering is best effort
        return f"<{type(value).__name__}>"


def _result_changes(
    before: ResultSummary | None, after: ResultSummary, with_missing: bool = False
) -> list[str]:
    """Explain how a replay's result differs in kind from the recorded result."""
    if before is None:
        return []
    reasons = []
    if before.kind != after.kind:
        reasons.append(f"result changed from {before.type_name} to {after.type_name}")
    elif before.kind == "scalar" and before.numeric and not after.numeric:
        # Scalars only: a frame or Series result carries its input's dtype through, so
        # its going non-numeric is expected. A numeric reduction turning into a string
        # (``sum`` concatenating) is the smell this catches.
        change = (
            ""
            if before.type_name == after.type_name
            else f" ({before.type_name} -> {after.type_name})"
        )
        reasons.append(f"result no longer numeric{change}")
    if after.all_na and not before.all_na and not with_missing:
        reasons.append("result is all missing")
    return reasons
