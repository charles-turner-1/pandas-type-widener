"""The dtype grid: naming, casting, building, and family collapse."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pandas_type_widener.dtypes import (
    collapse,
    default_grid,
    describe_index,
    describe_series,
    index_grid,
)

GRID = default_grid()
UNIVERSE = {s.name: s.family for s in GRID}


def test_grid_names_are_unique():
    """Every spec has its own name."""
    assert len(UNIVERSE) == len(GRID)


@pytest.mark.parametrize("spec", GRID, ids=lambda s: s.name)
def test_cast_from_ints_is_named_as_the_spec(spec):
    """A cast column describes itself with the spec's own name, keeping index and name."""
    original = pd.Series([1, 2, 3], index=[10, 20, 30], name="a")
    cast = spec.cast(original)
    assert cast is not None
    assert describe_series(cast) == spec.name
    assert list(cast.index) == [10, 20, 30] and cast.name == "a"


@pytest.mark.parametrize("spec", GRID, ids=lambda s: s.name)
def test_built_columns_are_named_as_the_spec(spec):
    """Drawn values build columns of the spec's dtype, with missing where nullable."""

    @settings(max_examples=5, deadline=None)
    @given(st.lists(spec.elements, min_size=2, max_size=4))
    def check(values):
        if spec.nullable:
            values = [None, *values]
        built = spec.build(values, pd.RangeIndex(len(values)), "a")
        assert describe_series(built) == spec.name
        assert bool(built.isna().iloc[0]) is spec.nullable

    check()


def test_cast_keeps_values():
    """Casting changes the dtype, not the data."""
    by_name = {s.name: s for s in GRID}
    original = pd.Series([1, 2, 3])
    assert list(by_name["float64"].cast(original)) == [1.0, 2.0, 3.0]
    assert list(by_name["object[str]"].cast(original)) == ["1", "2", "3"]
    assert list(by_name["category[int]"].cast(original).cat.categories) == [1, 2, 3]


def test_impossible_cast_returns_none():
    """Strings that aren't numbers don't cast to int."""
    by_name = {s.name: s for s in GRID}
    assert by_name["int64"].cast(pd.Series(["a", "b"])) is None


def test_cast_does_not_warn():
    """Our conversions' warnings never reach the user's warning summary."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for spec in GRID:
            spec.cast(pd.Series(["x", "y"], dtype=object))


@pytest.mark.parametrize("spec", index_grid(), ids=lambda s: s.name)
def test_index_grid_names(spec):
    """Each index spec builds an index of the right length, named as the spec."""
    index = spec.make(4)
    assert len(index) == 4 and describe_index(index) == spec.name


def test_describe_object_columns_by_content():
    """Object columns are named by what they hold."""
    assert describe_series(pd.Series(["a", None], dtype=object)) == "object[str]"
    assert describe_series(pd.Series(["a", 1], dtype=object)) == "object[mixed]"


def test_collapse_any():
    """Everything tried is 'any'."""
    assert collapse(UNIVERSE, UNIVERSE) == ["any"]


def test_collapse_groups_families_and_leftovers():
    """Whole groups, then whole families, then single names, in grid order."""
    numeric = [n for n, f in UNIVERSE.items() if f in ("integer", "float")]
    assert collapse(numeric, UNIVERSE) == ["numeric"]
    ints = [n for n, f in UNIVERSE.items() if f == "integer"]
    assert collapse([*ints, "float64", "bool"], UNIVERSE) == [
        "integer",
        "bool",
        "float64",
    ]
    assert collapse([], UNIVERSE) == []
    assert collapse(["not-a-dtype"], UNIVERSE) == []


def test_collapse_judges_families_against_what_was_tried():
    """A family is whole when every *tried* member is in, even if the grid has more."""
    universe = {"int64": "integer", "float64": "float", "str": "string"}
    assert collapse(["int64", "float64"], universe) == ["numeric"]


def test_str_spec_present_on_new_pandas():
    """pandas >= 2.3 has the ``str`` dtype; the grid tries it."""
    try:
        pd.StringDtype(na_value=np.nan)
    except TypeError:
        pytest.skip("pandas < 2.3")
    assert "str" in UNIVERSE
