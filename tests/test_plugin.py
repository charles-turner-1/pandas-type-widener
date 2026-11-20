"""End to end: the plugin driven by pytester over small suites that use toylib."""

import pytest

import toylib
from pandas_type_widener.targets import WRAPPED_ATTR

SUM = """
import pandas as pd
import toylib

def test_sum():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert toylib.sum_col(df, "a") == 6
"""

FMT = """
import pandas as pd
import toylib

def test_fmt():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert toylib.fmt_col(df, "a") == ["<1>", "<2>", "<3>"]
"""


def test_inert_without_targets(pytester):
    """Installed but not asked to probe, the plugin does nothing."""
    pytester.makepyfile(SUM)
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    assert "pandas type widening" not in result.stdout.str()


def test_terminal_report(widen):
    """The summary prints after the run; the tests' own outcomes are untouched."""
    run = widen(SUM)
    run.result.assert_outcomes(passed=1)
    run.result.stdout.fnmatch_lines(
        [
            "*= pandas type widening =*",
            "toylib.sum_col  df['a']  tested int64  (1 call site)",
            "  call     accepts  numeric, bool, complex",
            "*rejects*TypeError*",
            "note: 'accepts' means*",
        ]
    )


def test_widens_int_to_numeric(widen):
    """Tested on int64 alone, sum_col is found to take any numeric column."""
    s = widen(SUM).summary("toylib.sum_col", "df['a']")
    assert s["tested"] == ["int64"]
    assert s["accepts"][0] == "numeric"
    assert (
        "datetime64[ns]" in s["call"]
        and s["call"]["datetime64[ns]"]["status"] == "rejected"
    )


def test_oracles_disagree_on_value_assertions(widen):
    """Floats format fine (call oracle) but fail the test's exact-string assertion."""
    s = widen(FMT).summary("toylib.fmt_col", "df['a']")
    assert s["accepts"] == ["any"]
    assert s["test"]["float64"]["status"] == "failed"
    assert s["test"]["float64"]["detail"].startswith("AssertionError")
    assert s["test"]["int8"]["status"] == "passed"


def test_call_raised_in_rerun(widen):
    """The test oracle tells a raising target apart from a failing assertion."""
    s = widen(SUM).summary("toylib.sum_col", "df['a']")
    assert s["test"]["datetime64[ns]"]["status"] == "call-raised"


def test_call_oracle_only(widen):
    """``--widen-oracle call`` skips the re-runs."""
    run = widen(SUM, "--widen-oracle", "call")
    assert run.cells and all(c["test_oracle"] is None for c in run.cells)


def test_test_oracle_only(widen):
    """``--widen-oracle test`` keeps only the re-run verdicts."""
    run = widen(SUM, "--widen-oracle", "test")
    assert all(c["call"] is None and c["test_oracle"] for c in run.cells)


def test_failing_tests_are_not_probed(widen):
    """Only passing tests are probed, and a failure stays a failure."""
    run = widen(
        SUM
        + "\n\ndef test_broken():\n    toylib.fmt_col(pd.DataFrame({'a': [1]}), 'a')\n    assert False\n"
    )
    run.result.assert_outcomes(passed=1, failed=1)
    assert {c["test"].split("::")[1] for c in run.cells} == {"test_sum"}


def test_expected_exceptions_are_not_probed(widen):
    """A call the test expects to raise has nothing to widen."""
    source = """
import pandas as pd
import pytest
import toylib

def test_rejects_str():
    with pytest.raises(TypeError):
        toylib.double_ints(pd.DataFrame({"a": ["x"]}), "a")
"""
    run = widen(source)
    run.result.assert_outcomes(passed=1)
    assert run.cells == []
    run.result.stdout.fnmatch_lines(["no passing test called *"])


def test_outermost_call_only(widen):
    """A target that calls another target is reported once, as itself."""
    run = widen(
        "import pandas as pd\nimport toylib\n\ndef test_outer():\n    assert toylib.outer(pd.DataFrame({'a': [1]}), 'a') == 2\n"
    )
    assert {target for target, _ in run.labels()} == {"toylib.outer"}


def test_from_import_in_test_module(widen):
    """``from toylib import f`` in the test module is still seen."""
    run = widen(
        "import pandas as pd\nfrom toylib import sum_col\n\ndef test_it():\n    assert sum_col(pd.DataFrame({'a': [1]}), 'a') == 1\n"
    )
    assert ("toylib.sum_col", "df['a']") in run.labels()


def test_methods_and_reexports(widen):
    """Methods are targets, and a re-exported function keeps its public name."""
    source = """
import pandas as pd
import toylib

def test_it():
    left = pd.DataFrame({"k": [1, 2], "x": [1, 2]})
    right = pd.DataFrame({"k": [1, 2], "y": [3, 4]})
    assert len(toylib.merge_on(left, right, "k")) == 2
    assert toylib.Summariser("max").run(left, "x") == 2
"""
    labels = widen(source).labels()
    assert {
        ("toylib.merge_on", "left['k']"),
        ("toylib.merge_on", "right['k']"),
    } <= labels
    assert ("toylib.Summariser.run", "df['x']") in labels


def test_hypothesis_tests_get_the_call_oracle_only(widen):
    """A @given test is probed by the call oracle; its re-runs wouldn't line up."""
    source = """
import pandas as pd
import toylib
from hypothesis import given, settings, strategies as st

@settings(max_examples=5, database=None)
@given(st.lists(st.integers(0, 9), min_size=1, max_size=3))
def test_sum(values):
    assert toylib.sum_col(pd.DataFrame({"a": values}), "a") == sum(values)
"""
    run = widen(source)
    run.result.assert_outcomes(passed=1)
    assert run.cells and all(c["test_oracle"] is None for c in run.cells)


def test_max_sites(widen):
    """At most N call sites per target and schema are probed."""
    source = "import pandas as pd\nimport pytest\nimport toylib\n\n@pytest.mark.parametrize('n', range(4))\ndef test_sum(n):\n    toylib.sum_col(pd.DataFrame({'a': [n, 1]}), 'a')\n"
    run = widen(source, "--widen-max-sites", "2", "--widen-oracle", "call")
    run.result.assert_outcomes(passed=4)
    assert run.summary("toylib.sum_col", "df['a']")["sites"] == 2


def test_dtype_filter(widen):
    """``--widen-dtypes`` narrows the grid by name or family."""
    from pandas_type_widener.dtypes import default_grid

    run = widen(SUM, "--widen-dtypes", "float,int64", "--widen-oracle", "call")
    floats = {s.name for s in default_grid() if s.family == "float"}
    assert {c["dtype"] for c in run.cells if c["axis"] == "column"} == floats | {
        "int64"
    }
    assert {c["dtype"] for c in run.cells if c["axis"] == "index"} == {
        "int64",
        "float64",
    }


def test_joint_option(widen):
    """``--widen-joint`` adds the all-columns axis."""
    source = "import pandas as pd\nimport toylib\n\ndef test_sum():\n    toylib.sum_col(pd.DataFrame({'a': [1], 'b': [2]}), 'a')\n"
    run = widen(source, "--widen-joint", "--widen-oracle", "call")
    assert ("toylib.sum_col", "df[*]") in run.labels()


def test_ini_targets(pytester, widen):
    """The ``widen_targets`` ini option enables the plugin like ``--widen``."""
    pytester.makeini(
        "[pytest]\nwiden_targets =\n    toylib.sum_col\nwiden_oracle = call\n"
    )
    run = widen(SUM, targets=())
    assert run.labels() == {
        ("toylib.sum_col", "df['a']"),
        ("toylib.sum_col", "df.index"),
    }
    assert all(c["test_oracle"] is None for c in run.cells)


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (("no_such_package_xyz",), "*can't import*"),
        (("toylib.nothing_here",), "*matched no functions*"),
    ],
)
def test_usage_errors(widen, targets, message):
    """Targets that match nothing stop the run with a usage error."""
    run = widen(SUM, targets=targets)
    assert run.result.ret == pytest.ExitCode.USAGE_ERROR
    run.result.stderr.fnmatch_lines([message])


def test_wrappers_removed_after_run(widen):
    """Nothing stays wrapped once the session ends."""
    widen(SUM)
    assert not hasattr(toylib.sum_col, WRAPPED_ATTR)


def test_xdist(widen):
    """Under pytest-xdist, workers' results reach the controller's report."""
    pytest.importorskip("xdist")
    run = widen(
        SUM + FMT.replace("import pandas as pd\nimport toylib\n", ""),
        "-n",
        "2",
        "--widen-oracle",
        "call",
        subprocess=True,
    )
    run.result.assert_outcomes(passed=2)
    assert {t for t, _ in run.labels()} == {"toylib.sum_col", "toylib.fmt_col"}
    run.result.stdout.fnmatch_lines(["*= pandas type widening =*"])
