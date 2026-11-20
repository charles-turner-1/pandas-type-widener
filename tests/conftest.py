"""Shared fixtures: a pytester runner that drives the plugin against ``toylib``."""

import json
from pathlib import Path

import pytest

TESTS = Path(__file__).parent


class Run:
    """The outcome of one inner pytest run under the plugin."""

    def __init__(self, result, report_path):
        self.result = result
        self.data = (
            json.loads(report_path.read_text()) if report_path.exists() else None
        )

    @property
    def cells(self):
        """Every cell result in the JSON report."""
        return self.data["cells"]

    def summary(self, target, label):
        """The summary for one target argument, by its report label (``df['a']``)."""
        matches = [
            s
            for s in self.data["summaries"]
            if s["target"] == target and s["label"] == label
        ]
        assert matches, f"no summary for {target} {label}: " + ", ".join(
            f"{s['target']} {s['label']}" for s in self.data["summaries"]
        )
        return matches[0]

    def labels(self):
        """Every (target, label) pair that was summarised."""
        return {(s["target"], s["label"]) for s in self.data["summaries"]}


@pytest.fixture
def widen(pytester, monkeypatch):
    """Run a test-module source under ``--widen toylib`` and read back the JSON report.

    Extra arguments go to pytest; ``subprocess=True`` runs it out of process (for xdist).
    """
    pytester.syspathinsert(TESTS)
    monkeypatch.setenv("PYTHONPATH", str(TESTS))

    def run(source, *args, targets=("toylib",), subprocess=False):
        pytester.makepyfile(test_suite=source)
        report = pytester.path / "widen.json"
        argv = [a for t in targets for a in ("--widen", t)]
        argv += ["--widen-report", str(report), "-p", "no:cacheprovider", *args]
        runner = pytester.runpytest_subprocess if subprocess else pytester.runpytest
        return Run(runner(*argv), report)

    return run
