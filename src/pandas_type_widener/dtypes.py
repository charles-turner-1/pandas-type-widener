"""The dtype grid: which column and index types to try, and how to make them.

Each :class:`DtypeSpec` knows three things about its dtype:

- how to **cast** an existing column to it. This is the first choice, because it keeps
  the test's own values and so changes *only* the dtype. Join keys still match, a
  filter still selects the same rows.
- a Hypothesis strategy for **drawing** fresh values, used when the cast is impossible
  (``"a"`` can't become an int).
- whether it can hold **missing** values, which gets a separate check.

Specs belong to families (integer, float, string and so on). Families roll up into the
groups used to summarise results, ``numeric`` and ``temporal``. ``any`` means every
spec that was tried.
"""

from __future__ import annotations

import datetime as dt
import functools
import string
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from hypothesis import strategies as st

__all__ = [
    "FAMILY_ORDER",
    "GROUPS",
    "DtypeSpec",
    "IndexSpec",
    "collapse",
    "default_grid",
    "describe_dtype",
    "describe_index",
    "describe_series",
    "index_grid",
    "is_missing",
]

FAMILY_ORDER = (
    "bool",
    "integer",
    "float",
    "complex",
    "string",
    "categorical",
    "datetime",
    "timedelta",
    "period",
    "interval",
    "multi",
    "object",
)
GROUPS: dict[str, tuple[str, ...]] = {
    "numeric": ("integer", "float"),
    "temporal": ("datetime", "timedelta", "period"),
}

_TEXT = st.text(alphabet=string.ascii_letters + string.digits, max_size=8)


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _str_dtype() -> Any | None:
    """Return pandas' ``str`` dtype (the pandas 3 default), where this pandas has it."""
    try:
        return pd.StringDtype(na_value=np.nan)
    except TypeError:  # pandas < 2.3
        return None


def is_missing(value: Any) -> bool:
    """Return whether a scalar is a missing-value marker (None, NaN, NA, NaT).

    Parameters
    ----------
    value : Any
        Any cell value.

    Returns
    -------
    bool
        True for missing markers; False for anything else, including list-likes.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def describe_dtype(dtype: Any, values: Any = None) -> str:
    """Name a dtype the way the grid names it.

    ``str(dtype)`` isn't enough. It calls both string storages ``string`` and every
    categorical ``category``. Object columns are named by content: ``object[str]`` or
    ``object[mixed]``.

    Parameters
    ----------
    dtype : dtype-like
        The dtype.
    values : array-like, optional
        The values, which decide what an object dtype holds.

    Returns
    -------
    str
        The grid name.
    """
    if isinstance(dtype, pd.StringDtype):
        if getattr(dtype, "na_value", pd.NA) is not pd.NA:
            return "str"
        return f"string[{dtype.storage}]"
    if isinstance(dtype, pd.CategoricalDtype):
        cats = dtype.categories
        if cats is None:
            return "category"
        if pd.api.types.is_integer_dtype(cats.dtype):
            return "category[int]"
        if pd.api.types.is_string_dtype(cats.dtype):
            return "category[str]"
        return f"category[{describe_dtype(cats.dtype)}]"
    if pd.api.types.is_object_dtype(dtype):
        if values is None:
            return "object"
        kind = pd.api.types.infer_dtype(values, skipna=True)
        return "object[str]" if kind in ("string", "empty") else "object[mixed]"
    return str(dtype)


def describe_series(series: pd.Series) -> str:
    """Name a column's dtype the way the grid names it.

    Parameters
    ----------
    series : pandas.Series
        The column.

    Returns
    -------
    str
        The grid name.
    """
    return describe_dtype(series.dtype, series)


def describe_index(index: pd.Index) -> str:
    """Name an index the way the index grid names it.

    Parameters
    ----------
    index : pandas.Index
        The index.

    Returns
    -------
    str
        The grid name.
    """
    if isinstance(index, pd.MultiIndex):
        return "MultiIndex"
    if isinstance(index, pd.RangeIndex):
        return "RangeIndex"
    return describe_dtype(index.dtype, index)


@dataclass(frozen=True)
class DtypeSpec:
    """One cell of the dtype grid.

    Attributes
    ----------
    name : str
        Grid name, as :func:`describe_dtype` would give it.
    family : str
        One of :data:`FAMILY_ORDER`.
    dtype : dtype-like
        What to build columns with.
    elements : SearchStrategy
        Strategy for one non-missing value.
    nullable : bool
        Whether the dtype can hold missing values.
    caster : callable, optional
        Custom cast; default ``series.astype(dtype)``.
    reshape : callable, optional
        Applied to drawn values before building a column from them.
    """

    name: str
    family: str
    dtype: Any
    elements: st.SearchStrategy[Any]
    nullable: bool = True
    caster: Callable[[pd.Series], pd.Series] | None = None
    reshape: Callable[[Sequence[Any]], list[Any]] | None = None

    def cast(self, series: pd.Series) -> pd.Series | None:
        """Convert a column to this dtype, keeping its values where possible.

        Parameters
        ----------
        series : pandas.Series
            The column to convert.

        Returns
        -------
        pandas.Series or None
            The converted column (same index and name), or None if it can't convert.
        """
        try:
            with warnings.catch_warnings():
                # Our conversion's warnings (date parsing, say) are not the library's.
                warnings.simplefilter("ignore")
                out = self.caster(series) if self.caster else series.astype(self.dtype)
        except Exception:
            return None
        if len(out) != len(series):  # pragma: no cover - defensive
            return None
        return out

    def build(self, values: Sequence[Any], index: pd.Index, name: Any) -> pd.Series:
        """Build a column of this dtype from drawn values.

        Parameters
        ----------
        values : sequence
            Cell values; None means missing.
        index : pandas.Index
            Index for the new column.
        name : hashable
            Name for the new column.

        Returns
        -------
        pandas.Series
            The column.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cells = self.reshape(values) if self.reshape else list(values)
            return pd.Series(cells, dtype=self.dtype, index=index, name=name)


def _as_str(series: pd.Series, dtype: Any) -> pd.Series:
    values = [None if is_missing(v) else str(v) for v in series]
    return pd.Series(values, dtype=dtype, index=series.index, name=series.name)


def _mix(values: Sequence[Any]) -> list[Any]:
    """Make every other value a different kind from the rest, so the column really is mixed.

    Values at even positions set the kind. If the first one present is a string, the odd
    positions become non-strings (a string becomes its length); otherwise they become
    strings. Changing the type of every other value isn't enough on its own, because
    ``[None, 3, "a"]`` would come out as all strings.
    """
    out = list(values)
    anchor = next((v for v in out[0::2] if not is_missing(v)), None)
    want_str = not isinstance(anchor, str)
    for i in range(1, len(out), 2):
        value = out[i]
        if is_missing(value):
            continue
        if want_str and not isinstance(value, str):
            out[i] = str(value)
        elif not want_str and isinstance(value, str):
            out[i] = len(value)
    return out


def _mixed(series: pd.Series) -> pd.Series:
    """Keep the values, but change the type of every other one."""
    return pd.Series(_mix(series), dtype=object, index=series.index, name=series.name)


def _categorical_of_str(series: pd.Series) -> pd.Series:
    return _as_str(series, object).astype("category")


def _categorical_of_int(series: pd.Series) -> pd.Series:
    return series.astype("int64").astype("category")


def _to_datetime(series: pd.Series, unit: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        out = series
    elif pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
        series.dtype
    ):
        # Read numbers as days since the epoch: small integers then give distinct,
        # ordinary-looking dates instead of nanoseconds past 1970.
        out = pd.to_datetime(series.astype("float64"), unit="D")
    else:
        out = pd.to_datetime(series)
    if getattr(out.dtype, "tz", None) is not None:
        out = out.dt.tz_convert(None)
    return out.astype(f"datetime64[{unit}]")


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    return _to_datetime(series, "ns").dt.tz_localize("UTC")


def _to_boolean(series: pd.Series) -> pd.Series:
    try:
        return series.astype("boolean")
    except (
        TypeError,
        ValueError,
    ):  # only 0/1 cast directly; anything else by truthiness
        out = series.astype(bool).astype("boolean")
        return out.mask(series.isna().to_numpy())


def _to_timedelta(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series.dtype):
        return series.astype("timedelta64[ns]")
    return pd.to_timedelta(series.astype("float64"), unit="D").astype("timedelta64[ns]")


def _to_period(series: pd.Series) -> pd.Series:
    return _to_datetime(series, "ns").dt.to_period("D")


def _to_interval(series: pd.Series) -> pd.Series:
    left = series.astype("int64")
    array = pd.arrays.IntervalArray.from_arrays(left, left + 1)
    return pd.Series(array, index=series.index, name=series.name)


def _int_elements(dtype: Any) -> st.SearchStrategy[int]:
    info = np.iinfo(dtype)
    return st.integers(max(int(info.min), -1000), min(int(info.max), 1000))


def _float_elements(width: Literal[32, 64]) -> st.SearchStrategy[float]:
    return st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False, width=width)


_DATETIMES = st.datetimes(
    min_value=dt.datetime(1990, 1, 1), max_value=dt.datetime(2050, 1, 1)
)


def default_grid() -> list[DtypeSpec]:
    """Build the column dtype grid for this pandas, numpy and pyarrow.

    Returns
    -------
    list of DtypeSpec
        Every spec, ordered by family.
    """
    specs: list[DtypeSpec] = [
        DtypeSpec("bool", "bool", "bool", st.booleans(), nullable=False),
        DtypeSpec("boolean", "bool", "boolean", st.booleans(), caster=_to_boolean),
    ]
    for bits in (8, 16, 32, 64):
        specs.append(
            DtypeSpec(
                f"int{bits}",
                "integer",
                f"int{bits}",
                _int_elements(f"int{bits}"),
                False,
            )
        )
    for bits in (8, 16, 32, 64):
        specs.append(
            DtypeSpec(
                f"uint{bits}",
                "integer",
                f"uint{bits}",
                _int_elements(f"uint{bits}"),
                False,
            )
        )
    for bits in (8, 16, 32, 64):
        specs.append(
            DtypeSpec(
                f"Int{bits}", "integer", f"Int{bits}", _int_elements(f"int{bits}")
            )
        )
    for bits in (8, 16, 32, 64):
        specs.append(
            DtypeSpec(
                f"UInt{bits}", "integer", f"UInt{bits}", _int_elements(f"uint{bits}")
            )
        )
    specs += [
        DtypeSpec("float32", "float", "float32", _float_elements(32)),
        DtypeSpec("float64", "float", "float64", _float_elements(64)),
        DtypeSpec("Float32", "float", "Float32", _float_elements(32)),
        DtypeSpec("Float64", "float", "Float64", _float_elements(64)),
        DtypeSpec(
            "complex128",
            "complex",
            "complex128",
            st.complex_numbers(
                max_magnitude=1e3, allow_nan=False, allow_infinity=False
            ),
        ),
        DtypeSpec(
            "object[str]", "string", object, _TEXT, caster=lambda s: _as_str(s, object)
        ),
    ]
    str_dtype = _str_dtype()
    if str_dtype is not None:
        specs.append(
            DtypeSpec(
                "str",
                "string",
                str_dtype,
                _TEXT,
                caster=lambda s: _as_str(s, str_dtype),
            )
        )
    storages = ["python", "pyarrow"] if _has_pyarrow() else ["python"]
    for storage in storages:
        string_dtype = pd.StringDtype(storage)
        specs.append(
            DtypeSpec(
                f"string[{storage}]",
                "string",
                string_dtype,
                _TEXT,
                caster=functools.partial(_as_str, dtype=string_dtype),
            )
        )
    specs += [
        DtypeSpec(
            "category[str]",
            "categorical",
            "category",
            st.sampled_from(["a", "b", "c", "d"]),
            caster=_categorical_of_str,
        ),
        DtypeSpec(
            "category[int]",
            "categorical",
            "category",
            st.integers(0, 5),
            caster=_categorical_of_int,
        ),
        DtypeSpec(
            "datetime64[ns]",
            "datetime",
            "datetime64[ns]",
            _DATETIMES,
            caster=lambda s: _to_datetime(s, "ns"),
        ),
        DtypeSpec(
            "datetime64[us]",
            "datetime",
            "datetime64[us]",
            _DATETIMES,
            caster=lambda s: _to_datetime(s, "us"),
        ),
        DtypeSpec(
            "datetime64[ns, UTC]",
            "datetime",
            "datetime64[ns, UTC]",
            st.datetimes(
                min_value=dt.datetime(1990, 1, 1),
                max_value=dt.datetime(2050, 1, 1),
                timezones=st.just(dt.timezone.utc),
            ),
            caster=_to_datetime_utc,
        ),
        DtypeSpec(
            "timedelta64[ns]",
            "timedelta",
            "timedelta64[ns]",
            st.timedeltas(
                min_value=dt.timedelta(days=-1000), max_value=dt.timedelta(days=1000)
            ),
            caster=_to_timedelta,
        ),
        DtypeSpec(
            "period[D]",
            "period",
            pd.PeriodDtype("D"),
            st.dates(dt.date(1990, 1, 1), dt.date(2050, 1, 1)).map(
                lambda d: pd.Period(d, freq="D")
            ),
            caster=_to_period,
        ),
        DtypeSpec(
            "interval[int64, right]",
            "interval",
            pd.IntervalDtype("int64", closed="right"),
            st.tuples(st.integers(-100, 100), st.integers(1, 10)).map(
                lambda t: pd.Interval(t[0], t[0] + t[1])
            ),
            nullable=False,  # integer bounds can't hold NaN
            caster=_to_interval,
        ),
        DtypeSpec(
            "object[mixed]",
            "object",
            object,
            st.one_of(
                st.integers(-100, 100),
                st.floats(-100, 100, allow_nan=False),
                _TEXT,
                st.booleans(),
            ),
            caster=_mixed,
            reshape=_mix,
        ),
    ]
    if _has_pyarrow():
        import pyarrow as pa

        specs += [
            DtypeSpec(
                "bool[pyarrow]", "bool", pd.ArrowDtype(pa.bool_()), st.booleans()
            ),
            DtypeSpec(
                "int64[pyarrow]",
                "integer",
                pd.ArrowDtype(pa.int64()),
                _int_elements("int64"),
            ),
            DtypeSpec(
                "double[pyarrow]",
                "float",
                pd.ArrowDtype(pa.float64()),
                _float_elements(64),
            ),
        ]
    order = {family: i for i, family in enumerate(FAMILY_ORDER)}
    return sorted(specs, key=lambda s: order[s.family])


@dataclass(frozen=True)
class IndexSpec:
    """One index type to try.

    Attributes
    ----------
    name : str
        Grid name, as :func:`describe_index` would give it.
    family : str
        One of :data:`FAMILY_ORDER`.
    make : callable
        Builds an index of a given length.
    """

    name: str
    family: str
    make: Callable[[int], pd.Index]


def index_grid() -> list[IndexSpec]:
    """Build the index grid.

    Returns
    -------
    list of IndexSpec
        Every index spec, ordered by family.
    """
    str_dtype = _str_dtype()
    specs = [
        IndexSpec("RangeIndex", "integer", lambda n: pd.RangeIndex(n)),
        IndexSpec(
            "int64", "integer", lambda n: pd.Index(np.arange(n, dtype="int64") * 2 + 1)
        ),
        IndexSpec(
            "float64", "float", lambda n: pd.Index(np.arange(n, dtype="float64") + 0.5)
        ),
        IndexSpec(
            "object[str]",
            "string",
            lambda n: pd.Index([f"r{i}" for i in range(n)], dtype=object),
        ),
        IndexSpec(
            "datetime64[ns]",
            "datetime",
            lambda n: pd.date_range("2000-01-01", periods=n, freq="D", unit="ns"),
        ),
        IndexSpec(
            "datetime64[ns, UTC]",
            "datetime",
            lambda n: pd.date_range(
                "2000-01-01", periods=n, freq="D", tz="UTC", unit="ns"
            ),
        ),
        IndexSpec(
            "period[D]",
            "period",
            lambda n: pd.period_range("2000-01-01", periods=n, freq="D"),
        ),
        IndexSpec(
            "category[str]",
            "categorical",
            lambda n: pd.CategoricalIndex([f"c{i}" for i in range(n)]),
        ),
        IndexSpec(
            "MultiIndex",
            "multi",
            lambda n: pd.MultiIndex.from_arrays(
                [np.arange(n), [f"k{i % 2}" for i in range(n)]],
                names=["level_0", "level_1"],
            ),
        ),
    ]
    if str_dtype is not None:
        specs.insert(
            4,
            IndexSpec(
                "str",
                "string",
                lambda n: pd.Index([f"r{i}" for i in range(n)], dtype=str_dtype),
            ),
        )
    return specs


def collapse(names: Iterable[str], universe: dict[str, str]) -> list[str]:
    """Summarise a set of grid names as the largest families and groups it covers.

    Parameters
    ----------
    names : iterable of str
        The names to summarise, e.g. every accepted dtype.
    universe : dict of str to str
        Every name that was tried, mapped to its family. Family membership is judged
        against this, so "float" means every float dtype that was tried.

    Returns
    -------
    list of str
        ``["any"]``; or groups, then whole families, then leftover names, in grid order.
    """
    chosen = set(names) & set(universe)
    if not chosen:
        return []
    if chosen == set(universe) and len(universe) > 1:
        return ["any"]
    members: dict[str, set[str]] = {}
    for name, family in universe.items():
        members.setdefault(family, set()).add(name)
    full = {family for family, names_ in members.items() if names_ <= chosen}
    out: list[str] = []
    covered: set[str] = set()
    for group, families in GROUPS.items():
        present = [f for f in families if f in members]
        if len(present) > 1 and all(f in full for f in present):
            out.append(group)
            covered.update(present)
    for family in FAMILY_ORDER:
        if family in full and family not in covered:
            out.append(family)
            covered.add(family)
    out += [
        name for name in universe if name in chosen and universe[name] not in covered
    ]
    return out
