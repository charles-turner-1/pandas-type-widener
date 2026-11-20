# pandas-type-widener

A pytest plugin (`pytest11` entry point `pandas_type_widener`). It records calls that
passing tests make into named library targets, then replays them with DataFrame and
Series dtypes changed, to report which dtypes each argument really accepts. `README.md`
covers what it does. The module docstrings in `src/pandas_type_widener/` cover how.

## Layout

| Module | Role |
| --- | --- |
| `targets.py` | Patterns to callables; wrapping and rebinding; undo |
| `runtime.py` | Record/inject context, `CallSite`, `Slot`, outermost-only depth counter |
| `dtypes.py` | Dtype and index grids, cast/draw/build, naming, family `collapse` |
| `probe.py` | Axes × grid → `CellResult`s, with call and missing-value verdicts |
| `report.py` | Aggregate to `Summary`s; text and JSON output |
| `plugin.py` | Options, `pytest_runtest_call` wrapper, test-oracle re-runs, xdist relay |

`tests/toylib/` is a fake library whose dtype contracts are known. The end-to-end tests
in `tests/test_plugin.py` drive the plugin with `pytester` against it, using the
`widen` fixture in `tests/conftest.py`.

## Environment: pixi

Everything runs through pixi, as CI does (`setup-pixi`, `pixi run -e <env> test-cov`).

| Task | Command |
| --- | --- |
| Tests | `pixi run test` |
| Tests with coverage | `pixi run test-cov` |
| One file | `pixi run pytest tests/test_probe.py -q` |
| Lint / format / types | `pixi run ruff` / `pixi run black` / `pixi run mypy` |
| Hooks | `pixi run pre-commit` |

The default environment is the latest Python, with pyarrow and the dev tools. The other
environments pin the CI matrix. Use them when a change touches pandas versions or
optional dependencies:

- `test-py310` gets pandas 2, because pandas 3 needs Python 3.11+.
- `test-pandas2` is pandas 2 on Python 3.13.
- `test-nopyarrow` has no pyarrow.

Behaviour that differs between pandas 2 and 3 is mostly string dtypes (pandas 3's `str`
default) and datetime resolution (`us` by default in 3). Keep both working, and detect
features (see `dtypes._str_dtype`) rather than checking versions.

pyarrow is optional: without it the grid only loses its pyarrow cells. Nothing may
import it unconditionally.

## Conventions

- NumPy-style docstrings, checked by ruff `D` and `numpydoc-validation` (pre-commit).
- `mypy --strict` over `src`. pandas has no stubs here, so it is `Any`.
- Never change a test's own outcome, and never let the plugin's own warnings or errors
  leak into the user's run. Casts and builds run under `warnings.catch_warnings()`.
  Probe failures go to the "internal error" lines of the report.
- Reports must be deterministic: Hypothesis runs with `derandomize=True`.
