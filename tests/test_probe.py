"""The probe engine against toylib's known contracts, without pytest in the loop."""

import pandas as pd

import toylib
from pandas_type_widener import runtime
from pandas_type_widener.dtypes import default_grid, index_grid
from pandas_type_widener.probe import Prober, axes
from pandas_type_widener.report import render, summarise
from pandas_type_widener.targets import Installer, resolve

UNIVERSE = {s.name: s.family for s in default_grid()}
INTEGERS = {n for n, f in UNIVERSE.items() if f == "integer"}


def probe(fn, joint=False, examples=10):
    """Record the calls ``fn`` makes into toylib and probe them; return the summaries."""
    installer = Installer()
    installer.install(resolve(["toylib"]))
    ctx = runtime.TestContext("t", "record")
    try:
        with runtime.activate(ctx):
            fn()
    finally:
        installer.uninstall()
    prober = Prober(default_grid(), index_grid(), examples=examples, joint=joint)
    cells = [c.to_dict() for site in ctx.sites for c in prober.probe(site)]
    return {(s.target, s.label): s for s in summarise(cells)}, cells


def test_integer_only_contract():
    """A function that checks for integers accepts exactly the integer family."""
    summaries, _ = probe(lambda: toylib.double_ints(pd.DataFrame({"a": [1, 2]}), "a"))
    s = summaries["toylib.double_ints", "df['a']"]
    assert set(s.names("call", "accepted")) == INTEGERS
    assert set(s.names("call", "rejected")) == set(UNIVERSE) - INTEGERS
    assert all(
        s.call[n]["detail"].startswith("TypeError") for n in s.names("call", "rejected")
    )


def test_unused_column_and_index_accept_anything():
    """Columns and indexes the target never reads accept every dtype."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    summaries, _ = probe(lambda: toylib.sum_col(df, "a"))
    assert summaries["toylib.sum_col", "df['b']"].names("call", "accepted") == list(
        UNIVERSE
    )
    index = summaries["toylib.sum_col", "df.index"]
    assert index.names("call", "accepted") == list(index.universe)


def test_numeric_widening_and_string_degradation():
    """Tested on int64, ``sum`` works on any number; on strings it concatenates."""
    summaries, _ = probe(lambda: toylib.sum_col(pd.DataFrame({"a": [1, 2]}), "a"))
    s = summaries["toylib.sum_col", "df['a']"]
    accepted = set(s.names("call", "accepted"))
    assert {"float64", "Float64", "uint8", "Int64"} <= accepted
    assert s.call["object[str]"]["status"] == "degraded"
    assert "str" in s.call["object[str]"]["detail"]
    assert s.call["datetime64[ns]"]["status"] == "rejected"


def test_missing_values_break_int_cast():
    """``astype(int64)`` takes any number, until there's a missing value."""
    summaries, _ = probe(lambda: toylib.as_int(pd.DataFrame({"a": [1, 2, 3]}), "a"))
    s = summaries["toylib.as_int", "df['a']"]
    assert s.call["float64"]["status"] == "accepted"
    assert s.missing["float64"]["status"] == "rejected"
    assert s.missing["Int64"]["status"] == "rejected"
    assert "int64" not in s.missing  # non-nullable: no missing-values check


def test_missing_propagation_is_not_degradation():
    """Missing in, missing out isn't held against a dtype."""
    summaries, _ = probe(
        lambda: toylib.double_ints(pd.DataFrame({"a": [1, 2, 3]}), "a")
    )
    s = summaries["toylib.double_ints", "df['a']"]
    assert s.names("missing", "degraded") == []


def test_drawn_values_when_cast_impossible():
    """Strings that don't parse as numbers get Hypothesis-drawn values instead."""
    _, cells = probe(lambda: toylib.upper_col(pd.DataFrame({"a": ["x", "y"]}), "a"))
    sources = {c["dtype"]: c["source"] for c in cells if c["axis"] == "column"}
    assert sources["int64"] == "drawn" and sources["str"] == "cast"
    int_cell = next(c for c in cells if c["axis"] == "column" and c["dtype"] == "int64")
    assert int_cell["call"]["status"] == "rejected"
    assert (
        "AttributeError" in int_cell["call"]["detail"] and int_cell["call"]["example"]
    )


def test_index_axis():
    """Resampling needs a datetime-like index."""
    df = pd.DataFrame({"v": [1, 2, 3]}, index=pd.date_range("2000-01-01", periods=3))
    summaries, _ = probe(lambda: toylib.weekly(df))
    s = summaries["toylib.weekly", "df.index"]
    assert s.call["RangeIndex"]["status"] == "rejected"
    assert s.call["datetime64[ns, UTC]"]["status"] == "accepted"


def test_series_slot():
    """A Series argument is probed along its values and its index."""
    summaries, _ = probe(lambda: toylib.series_mean(pd.Series([1.0, 2.0])))
    s = summaries["toylib.series_mean", "s"]
    assert s.call["int8"]["status"] == "accepted"
    assert s.call["object[str]"]["status"] in ("rejected", "degraded")
    assert ("toylib.series_mean", "s.index") in summaries


def test_two_frame_slots():
    """Both frames of a merge are probed; key dtypes must be compatible."""
    left = pd.DataFrame({"k": [1, 2], "x": [1, 2]})
    right = pd.DataFrame({"k": [1, 2], "y": [3, 4]})
    summaries, _ = probe(lambda: toylib.merge_on(left, right, "k"))
    assert (
        summaries["toylib.merge_on", "left['k']"].call["datetime64[ns]"]["status"]
        == "rejected"
    )
    assert (
        summaries["toylib.merge_on", "right['y']"].call["str"]["status"] == "accepted"
    )


def test_joint_axis():
    """Joint mode casts every column at once to one representative per family."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    summaries, cells = probe(lambda: toylib.sum_col(df, "a"), joint=True)
    joint = summaries["toylib.sum_col", "df[*]"]
    assert joint.call["float64"]["status"] == "accepted"
    assert {c["dtype"] for c in cells if c["axis"] == "joint"} <= {
        "bool",
        "int64",
        "float64",
        "str",
        "object[str]",
        "category[str]",
        "datetime64[ns]",
    }


def test_unreplayable_site_yields_nothing():
    """A recorded call that no longer replays is skipped, not misreported."""
    calls = []

    def flaky(df):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("only works once")
        return df

    site = runtime.CallSite(
        "t",
        "flaky",
        0,
        flaky,
        (pd.DataFrame({"a": [1]}),),
        {},
        [runtime.Slot("df", "args", 0, "frame")],
        runtime.ResultSummary.of(None),
    )
    flaky(pd.DataFrame())  # use up the one good call
    assert list(Prober(default_grid(), index_grid(), 3).probe(site)) == []


def test_new_warnings_degrade():
    """A warning the baseline call didn't emit marks the dtype degraded."""
    import warnings

    def picky(df):
        if df["a"].dtype == "float64":
            warnings.warn(
                "floats are deprecated here", DeprecationWarning, stacklevel=2
            )
        return len(df)

    site = runtime.CallSite(
        "t",
        "picky",
        0,
        picky,
        (pd.DataFrame({"a": [1]}),),
        {},
        [runtime.Slot("df", "args", 0, "frame")],
        runtime.ResultSummary.of(1),
    )
    cells = {
        c.dtype: c
        for c in Prober(default_grid(), index_grid(), 3).probe(site)
        if c.axis == "column"
    }
    assert cells["float64"].call.status == "degraded"
    assert "DeprecationWarning" in cells["float64"].call.detail
    assert cells["float32"].call.status == "accepted"


def test_reports_are_deterministic():
    """Two runs give identical results: Hypothesis is derandomised."""

    def body():
        toylib.upper_col(pd.DataFrame({"a": ["x", "y", "z"]}), "a")

    assert probe(body)[1] == probe(body)[1]


def test_axes_listing():
    """A frame has one axis per column, then the index."""
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    assert [(a.kind, a.label, a.tested) for a in axes(df, joint=False)] == [
        ("column", "a", "int64"),
        ("column", "b", describe_b(df)),
        ("index", None, "RangeIndex"),
    ]


def describe_b(df):
    """``b``'s grid name: ``str`` on pandas 3, ``object[str]`` before."""
    from pandas_type_widener.dtypes import describe_series

    return describe_series(df["b"])


def test_render_mentions_every_group():
    """The text report has a header per group and the closing note."""
    summaries, _ = probe(lambda: toylib.double_ints(pd.DataFrame({"a": [1, 2]}), "a"))
    text = "\n".join(render(list(summaries.values())))
    assert "toylib.double_ints  df['a']  tested int64" in text
    assert "accepts  integer" in text and "rejects" in text
    assert "df.index  tested RangeIndex  (1 call site)  ->  accepts any" in text
    assert text.endswith("for.")


def test_example_budget_respected():
    """A drawn cell the target accepts tries at most ``examples`` inputs."""
    tried = []

    def count(df):
        tried.append(1)
        return len(df)

    site = runtime.CallSite(
        "t",
        "count",
        0,
        count,
        (pd.DataFrame({"a": ["x", "y"]}),),
        {},
        [runtime.Slot("df", "args", 0, "frame")],
        runtime.ResultSummary.of(2),
    )
    int64 = next(s for s in default_grid() if s.name == "int64")
    list(Prober([int64], [], examples=7).probe(site))
    assert 2 <= len(tried) <= 1 + 7  # the baseline, then at most the budget
