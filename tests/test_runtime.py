"""Recording: slots, snapshots, outermost-only, and injection."""

import pandas as pd
import pytest

import toylib
from pandas_type_widener import runtime
from pandas_type_widener.targets import Installer, resolve


@pytest.fixture
def wrapped():
    """Wrap all of toylib for one test."""
    installer = Installer()
    installer.install(resolve(["toylib"]))
    yield
    installer.uninstall()


def record(fn):
    """Run ``fn`` under a recording context and return the call sites."""
    ctx = runtime.TestContext("t", "record")
    with runtime.activate(ctx):
        fn()
    return ctx.sites


def test_slot_names():
    """Slots are named by parameter, by var-positional index, or by keyword."""
    df, s = pd.DataFrame({"a": [1]}), pd.Series([1])

    def f(x, y, *rest, **kw):
        return None

    slots = runtime.find_slots(f, (df, 1, s, df), {"other": s, "n": 3})
    assert [(sl.name, sl.kind) for sl in slots] == [
        ("x", "frame"),
        ("*rest[0]", "series"),
        ("*rest[1]", "frame"),
        ("other", "series"),
    ]


def test_slot_replace_does_not_mutate():
    """Replacing a slot returns new containers."""
    args, kwargs = (1, 2), {"k": 3}
    slot = runtime.Slot("k", "kwargs", "k", "frame")
    new_args, new_kwargs = slot.replace(args, kwargs, 9)
    assert new_kwargs == {"k": 9} and kwargs == {"k": 3} and new_args == args


def test_records_only_frame_calls(wrapped):
    """Calls without a DataFrame or Series aren't recorded, but still count as calls."""
    df = pd.DataFrame({"a": [1, 2]})
    sites = record(lambda: (toylib.Summariser("sum"), toylib.sum_col(df, "a")))
    assert [(s.target, s.ordinal) for s in sites] == [("toylib.sum_col", 0)]


def test_snapshot_is_isolated_from_later_mutation(wrapped):
    """Mutating the frame after the call doesn't change what was recorded."""
    df = pd.DataFrame({"a": [1, 2]})

    def body():
        toylib.sum_col(df, "a")
        df["a"] = ["x", "y"]

    (site,) = record(body)
    assert list(site.args[0]["a"]) == [1, 2]


def test_only_outermost_call_recorded(wrapped):
    """A target calling another target is one recorded call."""
    sites = record(lambda: toylib.outer(pd.DataFrame({"a": [1]}), "a"))
    assert [s.target for s in sites] == ["toylib.outer"]


def test_raising_call_recorded_as_raised(wrapped):
    """A call that raised is kept, marked, and the exception still propagates."""
    ctx = runtime.TestContext("t", "record")
    with runtime.activate(ctx), pytest.raises(TypeError):
        toylib.double_ints(pd.DataFrame({"a": ["x"]}), "a")
    assert ctx.sites[0].raised == "TypeError" and ctx.sites[0].result is None


def test_no_context_passes_through(wrapped):
    """Outside a context the wrapper is transparent."""
    assert toylib.sum_col(pd.DataFrame({"a": [1, 2]}), "a") == 3


def test_injection_swaps_the_named_call(wrapped):
    """Injection changes exactly the planned call and records that it got there."""
    df = pd.DataFrame({"a": [1, 2]})
    slot = runtime.Slot("df", "args", 0, "frame")
    plan = runtime.Injection(1, "toylib.sum_col", slot, lambda live: live * 10)
    ctx = runtime.TestContext("t", "inject", injection=plan)
    with runtime.activate(ctx):
        first = toylib.sum_col(df, "a")
        second = toylib.sum_col(df, "a")
    assert (first, second) == (3, 30) and plan.reached


def test_injection_target_mismatch(wrapped):
    """An injection whose ordinal lands on a different target doesn't apply."""
    slot = runtime.Slot("df", "args", 0, "frame")
    plan = runtime.Injection(0, "toylib.fmt_col", slot, lambda live: live * 10)
    ctx = runtime.TestContext("t", "inject", injection=plan)
    with runtime.activate(ctx):
        assert toylib.sum_col(pd.DataFrame({"a": [1]}), "a") == 1
    assert not plan.reached and "sum_col" in plan.mismatch


@pytest.mark.parametrize(
    ("value", "kind", "numeric", "all_na"),
    [
        (3, "scalar", True, False),
        (float("nan"), "scalar", True, True),
        ("x", "scalar", False, False),
        (True, "scalar", False, False),
        (None, "none", False, False),
        (pd.Series([1.0, None]), "series", True, False),
        (pd.Series([None, None], dtype=float), "series", True, True),
        (pd.DataFrame({"a": [1], "b": ["x"]}), "frame", False, False),
        (pd.Timedelta(1), "other", False, False),
    ],
)
def test_result_summary(value, kind, numeric, all_na):
    """Return values are summarised by kind, numeric-ness and missing-ness."""
    summary = runtime.ResultSummary.of(value)
    assert (summary.kind, summary.numeric, summary.all_na) == (kind, numeric, all_na)
