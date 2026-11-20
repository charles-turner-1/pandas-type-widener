"""Target resolution and wrapper installation."""

import sys
import types

import pytest

import toylib
from pandas_type_widener.targets import WRAPPED_ATTR, Installer, resolve


def names(patterns):
    """Display names of the targets ``patterns`` resolve to."""
    return [t.name for t in resolve(patterns)]


def test_package_pattern_finds_functions_and_methods():
    """A package name matches its public functions and public class methods."""
    found = names(["toylib"])
    assert {"toylib.sum_col", "toylib.fmt_col", "toylib.weekly"} <= set(found)
    assert {"toylib.Summariser.run", "toylib.Summariser.count"} <= set(found)
    assert not any(".__init__" in n or "._" in n.rsplit(".", 1)[-1] for n in found)


def test_reexported_function_takes_its_public_name():
    """A function defined in a private module is named where it's re-exported."""
    assert "toylib.merge_on" in names(["toylib"])
    assert "toylib._impl.merge_on" not in names(["toylib"])


def test_private_path_still_matches():
    """Its defining path matches too, but the display name stays public."""
    assert names(["toylib._impl.merge_on"]) == ["toylib.merge_on"]


def test_glob_pattern():
    """Globs match over dotted names."""
    assert names(["toylib.s*_col"]) == ["toylib.sum_col"]


def test_class_pattern_covers_its_methods():
    """A class path names every public method of the class."""
    assert names(["toylib.Summariser"]) == [
        "toylib.Summariser.count",
        "toylib.Summariser.run",
    ]


def test_patterns_deduplicate():
    """Overlapping patterns give each target once."""
    assert names(["toylib.sum_col", "toylib.sum*"]) == ["toylib.sum_col"]


@pytest.mark.parametrize("pattern", ["no_such_package_xyz", "*.foo"])
def test_unimportable_pattern_raises(pattern):
    """A pattern whose package can't be imported is an error, not an empty match."""
    with pytest.raises(ValueError):
        resolve([pattern])


def test_install_rebinds_imports_and_uninstall_restores():
    """Wrappers replace the target where defined and where imported, and come back out."""
    original_sum, original_run = toylib.sum_col, toylib.Summariser.__dict__["run"]
    original_count = toylib.Summariser.__dict__["count"]
    importer = types.ModuleType("fake_importer")
    importer.sum_col = toylib.sum_col  # as if it did `from toylib import sum_col`
    sys.modules["fake_importer"] = importer
    installer = Installer()
    try:
        installer.install(resolve(["toylib.sum_col", "toylib.Summariser"]))
        assert getattr(toylib.sum_col, WRAPPED_ATTR) is original_sum
        assert importer.sum_col is toylib.sum_col
        assert isinstance(toylib.Summariser.__dict__["count"], staticmethod)
        assert toylib.Summariser.count(toylib.pd.DataFrame({"a": [1]})) == 1
        assert toylib.Summariser.__dict__["run"] is not original_run
    finally:
        installer.uninstall()
        del sys.modules["fake_importer"]
    assert toylib.sum_col is original_sum
    assert importer.sum_col is original_sum
    assert toylib.Summariser.__dict__["run"] is original_run
    assert toylib.Summariser.__dict__["count"] is original_count


def test_wrapper_keeps_signature():
    """``functools.wraps`` keeps the wrapped signature, which names the slots."""
    import inspect

    installer = Installer()
    installer.install(resolve(["toylib.sum_col"]))
    try:
        assert list(inspect.signature(toylib.sum_col).parameters) == ["df", "col"]
    finally:
        installer.uninstall()
