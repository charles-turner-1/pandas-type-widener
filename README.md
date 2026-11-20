# pandas-type-widener

> [!WARNING]
> This is an **experiment**. I had the idea and talked the overall approach through with Claude, then had Claude
> write essentially all of it, very quickly. I haven't read the code, and I haven't really used it yet. It might
> look good (IDK), but it has not been drive-tested in any meaningful sense of the word. Claims about functionality
> in this README should be considered aspirational.
> Use at your own caution (whilst this warning is still up).

> [!NOTE]
> **This is not a pandas project.** It isn't affiliated with, endorsed by, or supported by pandas or its
> maintainers. It just probes code that takes pandas objects. It also isn't really a *package*, despite looking
> like one: it's an LLM-built prototype to find out whether the idea is worth pursuing.

A pytest plugin that finds out which pandas dtypes a library **really** accepts, using
the test suite you already have.

Libraries that take DataFrames rarely say which column dtypes they support. Their tests
usually try one dtype per column, so the contract stays implicit: tested on `int64`, but
does it work on `float32`? On nullable `Int64`? On categoricals, or pandas 3's `str`?
This plugin runs your suite as normal. It then replays every call your tests make into
the library, swapping the dtypes of the DataFrame and Series arguments, and reports what
each argument actually tolerates.

```console
$ pytest --widen toylib
...
============================= pandas type widening =============================
toylib.sum_col  df['a']  tested int64  (1 call site)
  call     accepts  numeric, bool, complex
           degrades string, timedelta (result changed from int64 to Timedelta; ...)
           rejects  categorical, datetime, period, interval, object (TypeError x8)
  test     passes   numeric, complex
           fails    bool, string, timedelta (AssertionError x8)
           raises   categorical, datetime, period, interval, object (TypeError x8)
toylib.sum_col  df['b']  tested str  (1 call site)  ->  accepts any
toylib.weekly  df.index  tested datetime64[us]  (1 call site)
  call     accepts  temporal
           rejects  numeric, string, categorical, multi (TypeError x7)
```

So `sum_col` was tested with `int64` only, but it takes any numeric column. Strings
don't raise, but they concatenate instead of adding. Column `b` is never read.

## Usage

```console
pytest --widen mylib                      # every public function and method in mylib
pytest --widen 'mylib.api.*'              # glob over dotted names
pytest --widen mylib.frame.Summary.run    # one method
pytest --widen mylib --widen-report widen.json
```

The plugin is installed as a `pytest11` entry point, but does nothing until you name a
target. Targets can also go in configuration:

```toml
[tool.pytest.ini_options]
widen_targets = ["mylib"]
widen_oracle = "call"
```

| Option | Default | |
| --- | --- | --- |
| `--widen TARGET` / `widen_targets` | | Package, module, class or function path, or a glob over them. Repeatable. |
| `--widen-report PATH` | | Also write every result and summary as JSON. |
| `--widen-oracle {call,test,both}` | `both` | How each dtype is judged (see below). |
| `--widen-dtypes NAMES` | whole grid | Comma-separated dtype names or families (`float,int64,string`). |
| `--widen-examples N` | 10 | Hypothesis examples per randomised check. |
| `--widen-max-sites N` | 3 | Call sites probed per target and argument schema. |
| `--widen-joint` | off | Also cast all of a frame's columns at once. |

## How it works

**Discovery is done at runtime.** It does not read the test source. The target functions
are wrapped for the session: on the module or class that defines them, and wherever they
were already imported. Every call a *passing* test makes with a DataFrame or Series
argument is recorded, along with a snapshot of its arguments. Only the outermost call
counts, so a library function that calls another of its own functions is recorded once.
Calls that raised (a `pytest.raises` test) are not probed.

**Each argument is varied along axes:**

- each column, one at a time, keeping the others as they were;
- a Series' values;
- the index;
- optionally, all columns together.

**Each axis is crossed with a dtype grid:**

- `bool` and `boolean`
- numpy and nullable ints and uints
- floats and nullable floats
- `complex128`
- `object` strings, `str`, `string[python]` and `string[pyarrow]`
- categoricals of str and of int
- datetimes (ns, us, tz-aware)
- `timedelta`, `period`, `interval`
- mixed-object columns
- pyarrow-backed types

**Values come from the test where possible.** A column is first *cast* to the new dtype,
so only the dtype changes: join keys still match, filters still select the same rows.
When the values can't convert (`"a"` won't become an int), Hypothesis draws fresh values
instead. Hypothesis also picks where to put missing values, in a separate check for each
nullable dtype. Runs are derandomised, so reports don't change between runs.

### Two oracles, reported side by side

- **call** replays the target directly with the changed argument.
  - *accepted*: it returned.
  - *degraded*: it returned, but with something off:
    - it emitted a warning the original call didn't;
    - a numeric scalar result became non-numeric (`sum` of strings concatenates);
    - the result changed kind (a number became a `Timedelta`);
    - the result came back all-missing.
  - *rejected*: it raised.
- **test** re-runs the whole test with the changed argument swapped into the same call.
  - *passed*.
  - *failed*: the call worked, but a later assertion did not.
  - *call-raised*: the target raised.

The two oracles disagree in useful ways. `assert fmt(df) == ["<1>", "<2>"]` fails on
floats (`"<1.0>"`) even though `fmt` handles them fine. The call oracle says the dtype
works, and the test oracle says this test's expectations don't carry over to it.

**"Accepts" means "didn't raise", not "correct".** An f-string accepts anything. The
degraded checks catch the common silent failures, not all of them.

### JSON report

`--widen-report` writes `{"summaries": [...], "cells": [...], "errors": [...]}`. Each cell
records:

- the test, target, call ordinal, argument and axis;
- the dtype the test used, and the dtype tried;
- where the values came from: `cast`, `drawn` or `fixed`;
- the verdict of each oracle, with the exception and a rendering of the failing input.

A summary gives, for each target argument and axis, each dtype's worst verdict across
its call sites, collapsed to families (`accepts`, `degrades`, `rejects`).

## Limitations

- Frames nested inside lists or dicts are not probed. Neither are targets reached
  through references the plugin can't rebind, such as registries or
  `functools.partial`.
- The test oracle reuses the test's function-scoped fixture values on every re-run. A
  test that mutates its fixtures can mislead it. It is skipped for Hypothesis tests,
  whose call order changes between runs.
- Results under pytest-xdist are gathered on the controller, but `--widen-max-sites`
  applies per worker.
- Async functions are not wrapped.

## Planned

- Generating real pytest items per cell (record in one pass, expand in a second), so
  each dtype gets its own pass/fail in pytest's output.
- Discovering targets automatically with `sys.monitoring`.
- Turning a report into type hints or a docs table.

## Development

The environment is [pixi](https://pixi.sh): `pixi run test`, `pixi run ruff`,
`pixi run mypy`, `pixi run pre-commit`. See `CLAUDE.md` for the environment matrix.
