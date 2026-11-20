"""A private module: its function is re-exported, and reported, as ``toylib.merge_on``."""

import pandas as pd


def merge_on(left, right, key):
    """Merge two frames on ``key``; the key dtypes must be compatible."""
    return pd.merge(left, right, on=key)
