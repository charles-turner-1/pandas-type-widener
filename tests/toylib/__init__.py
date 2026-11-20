"""A toy pandas-consuming library whose dtype contracts are known in advance.

The plugin's end-to-end tests run small suites against it and check that the inferred
contracts match these.
"""

import pandas as pd

from ._impl import merge_on

__all__ = [
    "Summariser",
    "as_int",
    "double_ints",
    "fmt_col",
    "merge_on",
    "outer",
    "series_mean",
    "sum_col",
    "upper_col",
    "weekly",
]


def sum_col(df, col):
    """Sum a column: numeric and bool work; strings concatenate; dates raise."""
    return df[col].sum()


def fmt_col(df, col):
    """Format every value: takes anything."""
    return [f"<{v}>" for v in df[col]]


def double_ints(df, col):
    """Double an integer column; refuses anything else outright."""
    if not pd.api.types.is_integer_dtype(df[col].dtype):
        raise TypeError(f"{col!r} must be an integer column, got {df[col].dtype}")
    return df[col] * 2


def as_int(df, col):
    """Cast to int64: any numeric works, until a missing value turns up."""
    return df[col].astype("int64")


def upper_col(df, col):
    """Upper-case a column of strings via the ``.str`` accessor."""
    return df[col].str.upper()


def weekly(df):
    """Resample to weeks: needs a datetime-like index."""
    return df.resample("7D").sum()


def series_mean(s):
    """Mean of a Series."""
    return s.mean()


def outer(df, col):
    """Call another target internally: only this outer call should be recorded."""
    return sum_col(df, col) + 1


class Summariser:
    """A class whose methods are targets too."""

    def __init__(self, how="sum"):
        self.how = how

    def run(self, df, col):
        """Reduce a column with the configured method."""
        return getattr(df[col], self.how)()

    @staticmethod
    def count(df):
        """Count rows."""
        return len(df)
