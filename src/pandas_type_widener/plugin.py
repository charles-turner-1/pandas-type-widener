"""The pytest plugin: options, recording during tests, probing, and the report.

The plugin is loaded automatically once the package is installed, and does nothing
until ``--widen TARGET`` (or the ``widen_targets`` ini option) names something to probe.
Then, for every test that **passes**:

1. the test runs as normal while calls into the targets are recorded;
2. each recorded call that passed a DataFrame or Series is replayed across the dtype grid
   (the call oracle);
3. optionally the test itself is re-run once per cell with the changed argument swapped
   in (the test oracle).

Probing happens inside the test's call phase, after the test body, so its fixtures are
still live for re-runs. The test's own outcome is never changed. Under pytest-xdist each
worker sends its results to the controller as ``user_properties``, and the controller
reports.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Generator
from typing import Any

import pytest

from . import report, runtime
from .dtypes import default_grid, index_grid
from .probe import Prober, Verdict
from .runtime import Injection, TestContext
from .targets import Installer, resolve

__all__ = ["TypeWidener"]

PLUGIN_NAME = "pandas-type-widener-session"
_PROPERTY = "pandas_type_widener"
_ORACLES = ("call", "test", "both")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--widen*`` options and their ini equivalents.

    Parameters
    ----------
    parser : pytest.Parser
        pytest's option parser.
    """
    group = parser.getgroup("pandas-type-widener", "pandas dtype widening")
    group.addoption(
        "--widen",
        action="append",
        default=[],
        dest="widen_targets",
        metavar="TARGET",
        help="Probe calls into TARGET: a package, module, class or function path, or a glob "
        "over them (mylib, mylib.api.*). Repeatable. Enables the plugin.",
    )
    group.addoption(
        "--widen-report",
        dest="widen_report",
        metavar="PATH",
        help="Also write every result, and the summaries, to PATH as JSON.",
    )
    group.addoption(
        "--widen-oracle",
        dest="widen_oracle",
        choices=_ORACLES,
        help="Judge dtypes by the call alone, by re-running the test, or both (default: both).",
    )
    group.addoption(
        "--widen-joint",
        dest="widen_joint",
        action="store_true",
        default=None,
        help="Also vary all of a frame's columns at once.",
    )
    group.addoption(
        "--widen-examples",
        dest="widen_examples",
        type=int,
        metavar="N",
        help="Hypothesis examples per randomised check (default: 10).",
    )
    group.addoption(
        "--widen-max-sites",
        dest="widen_max_sites",
        type=int,
        metavar="N",
        help="Probe at most N call sites per target and argument schema (default: 3).",
    )
    group.addoption(
        "--widen-dtypes",
        dest="widen_dtypes",
        metavar="NAMES",
        help="Comma-separated dtype names or families to try (default: the whole grid).",
    )
    parser.addini(
        "widen_targets", "Targets to probe, one per line.", type="linelist", default=[]
    )
    parser.addini("widen_oracle", "call, test or both.", default="both")
    parser.addini(
        "widen_joint", "Also vary all columns at once.", type="bool", default=False
    )
    parser.addini(
        "widen_examples", "Hypothesis examples per randomised check.", default="10"
    )
    parser.addini("widen_max_sites", "Call sites per target and schema.", default="3")
    parser.addini("widen_dtypes", "Dtype names or families to try.", default="")


def pytest_configure(config: pytest.Config) -> None:
    """Start the plugin session if any targets were named.

    Parameters
    ----------
    config : pytest.Config
        The run's configuration.
    """
    targets = config.getoption("widen_targets") or config.getini("widen_targets")
    if targets:
        config.pluginmanager.register(TypeWidener(config, list(targets)), PLUGIN_NAME)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove the target wrappers.

    Parameters
    ----------
    config : pytest.Config
        The run's configuration.
    """
    plugin = config.pluginmanager.get_plugin(PLUGIN_NAME)
    if isinstance(plugin, TypeWidener):
        plugin.installer.uninstall()
        config.pluginmanager.unregister(plugin)


def _option(config: pytest.Config, name: str) -> Any:
    value = config.getoption(name)
    return config.getini(name) if value is None else value


class TypeWidener:
    """One plugin session: the installed wrappers, the prober, and collected results.

    Parameters
    ----------
    config : pytest.Config
        The run's configuration.
    patterns : list of str
        The target patterns.
    """

    def __init__(self, config: pytest.Config, patterns: list[str]) -> None:
        self.config = config
        self.oracle = str(_option(config, "widen_oracle"))
        if self.oracle not in _ORACLES:
            raise pytest.UsageError(
                f"widen_oracle must be one of {_ORACLES}, not {self.oracle!r}"
            )
        self.max_sites = int(_option(config, "widen_max_sites"))
        grid, igrid = default_grid(), index_grid()
        wanted = {
            w.strip()
            for w in str(_option(config, "widen_dtypes") or "").split(",")
            if w.strip()
        }
        if wanted:
            grid = [s for s in grid if s.name in wanted or s.family in wanted]
            igrid = [s for s in igrid if s.name in wanted or s.family in wanted]
            if not grid and not igrid:
                raise pytest.UsageError(
                    f"--widen-dtypes {sorted(wanted)} matches no dtype"
                )
        self.prober = Prober(
            grid,
            igrid,
            examples=int(_option(config, "widen_examples")),
            joint=bool(_option(config, "widen_joint")),
        )
        try:
            targets = resolve(patterns)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
        if not targets:
            raise pytest.UsageError(f"--widen {patterns} matched no functions")
        self.targets = targets
        self.installer = Installer()
        self.installer.install(targets)
        self.is_worker = hasattr(config, "workerinput")
        self.cells: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.seen: dict[tuple[Any, ...], int] = {}
        self.unreplayed = 0
        self.seconds = 0.0

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: pytest.Item) -> Generator[None, Any, Any]:
        """Record target calls during the test; probe them if it passes."""
        ctx = TestContext(item.nodeid, "record")
        with runtime.activate(ctx):
            outcome = yield
        if ctx.sites:
            start = time.perf_counter()
            try:
                cells = self._probe(item, ctx)
            except Exception:
                self.errors.append(f"{item.nodeid}: {traceback.format_exc()}")
                cells = []
            self.seconds += time.perf_counter() - start
            if self.is_worker:
                item.user_properties.append((_PROPERTY, cells))
            else:
                self.cells.extend(cells)
        return outcome

    def _probe(self, item: pytest.Item, ctx: TestContext) -> list[dict[str, Any]]:
        rerun = None
        if self.oracle in ("test", "both") and not getattr(
            getattr(item, "obj", None), "is_hypothesis_test", False
        ):
            # A Hypothesis test calls the target once per example, so call ordinals
            # don't line up between runs; only the call oracle applies to it.
            def rerun(plan: Injection) -> Verdict:
                return _rerun(item, plan)

        cells: list[dict[str, Any]] = []
        for site in ctx.sites:
            if site.raised is not None:
                continue  # the test expected this call to fail; nothing to widen
            key = site.schema()
            if self.seen.get(key, 0) >= self.max_sites:
                continue
            self.seen[key] = self.seen.get(key, 0) + 1
            results = list(self.prober.probe(site, rerun))
            if not results:
                self.unreplayed += 1
            for result in results:
                if self.oracle == "test":
                    result.call = result.call_missing = None
                cells.append(result.to_dict())
        return cells

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Collect results that xdist workers attached to their reports."""
        if self.is_worker or report.when != "call":
            return
        for name, value in report.user_properties:
            if name == _PROPERTY and isinstance(value, list):
                self.cells.extend(value)

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        """Print the support summary."""
        if self.is_worker:
            return
        tr = terminalreporter
        tr.write_sep("=", "pandas type widening")
        summaries = report.summarise(self.cells)
        if not summaries:
            tr.write_line(
                f"no passing test called {', '.join(t.name for t in self.targets[:5])}"
                f"{' ...' if len(self.targets) > 5 else ''} with a DataFrame or Series"
            )
        for line in report.render(summaries):
            tr.write_line(line)
        if self.unreplayed:
            tr.write_line(
                f"{self.unreplayed} call site(s) skipped: the recorded call didn't replay"
            )
        if self.seconds:
            tr.write_line(f"probing took {self.seconds:.1f}s")
        for error in self.errors:
            tr.write_line(f"internal error: {error}", red=True)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Write the JSON report, if asked for."""
        path = self.config.getoption("widen_report")
        if path and not self.is_worker:
            report.write_json(path, self.cells, self.errors)


def _rerun(item: pytest.Item, plan: Injection) -> Verdict:
    """Run a test body again with ``plan`` applied, and judge the result."""
    ctx = TestContext(item.nodeid, "inject", injection=plan)
    failure: BaseException | None = None
    with runtime.activate(ctx):
        try:
            item.runtest()
        except (pytest.skip.Exception, pytest.xfail.Exception):
            return Verdict("skipped")
        except (Exception, pytest.fail.Exception) as exc:
            failure = exc
    if plan.mismatch is not None or not plan.reached:
        return Verdict("unmatched", plan.mismatch or "the re-run didn't reach the call")
    if failure is None:
        detail = (
            ""
            if plan.raised is None
            else f"the call raised {type(plan.raised).__name__}"
        )
        return Verdict("passed", detail)
    if plan.raised is not None:
        return Verdict("call-raised", _short(plan.raised))
    return Verdict("failed", _short(failure))


def _short(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    message = text[0][:160] if text else ""
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
