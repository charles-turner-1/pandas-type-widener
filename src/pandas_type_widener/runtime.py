"""Runtime state shared by the target wrappers, the recorder and the probe engine.

Every wrapped target funnels through :func:`call_target`. What happens there depends on
the :class:`TestContext` that is active, if any:

- no context (outside a test's call phase, or while probing): call straight through;
- a ``"record"`` context: call through, and remember the call if it passed a DataFrame
  or Series (a :class:`CallSite`);
- an ``"inject"`` context: swap a mutated argument into one specific call, then call
  through.

Only the *outermost* target call counts. A library function that calls another of the
library's own functions is one call from the test's point of view, so a depth counter
lets the inner calls pass straight through.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import inspect
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "CallSite",
    "Injection",
    "ResultSummary",
    "Slot",
    "TestContext",
    "activate",
    "call_target",
    "find_slots",
]

_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "pandas_type_widener_depth", default=0
)
_ACTIVE: TestContext | None = None


@dataclass(frozen=True)
class Slot:
    """Where a DataFrame or Series argument sits in a call.

    Attributes
    ----------
    name : str
        The parameter it binds to, for reports (``df``, ``*frames[1]``, ``arg0``).
    where : {"args", "kwargs"}
        Whether it was passed positionally or by keyword.
    key : int or str
        Position in ``args`` or key in ``kwargs``.
    kind : {"frame", "series"}
        What was passed.
    """

    name: str
    where: Literal["args", "kwargs"]
    key: int | str
    kind: Literal["frame", "series"]

    def get(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Return the slot's value from a call's arguments.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        Any
            The argument in this slot.
        """
        if self.where == "args":
            assert isinstance(self.key, int)
            return args[self.key]
        return kwargs[str(self.key)]

    def replace(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], value: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return copies of a call's arguments with this slot set to ``value``.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.
        value : Any
            The replacement argument.

        Returns
        -------
        tuple of (tuple, dict)
            The new ``args`` and ``kwargs``.
        """
        if self.where == "args":
            assert isinstance(self.key, int)
            new_args = list(args)
            new_args[self.key] = value
            return tuple(new_args), dict(kwargs)
        return tuple(args), {**kwargs, str(self.key): value}


def find_slots(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[Slot]:
    """Find the DataFrame and Series arguments of a call.

    Only top-level arguments count: a frame inside a list or dict is not probed.

    Parameters
    ----------
    func : callable
        The function being called; its signature names the slots.
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    list of Slot
        One per DataFrame or Series argument, in call order.
    """
    positional: list[str] = []
    varargs: str | None = None
    try:
        for param in inspect.signature(func).parameters.values():
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                positional.append(param.name)
            elif param.kind is param.VAR_POSITIONAL:
                varargs = param.name
    except (TypeError, ValueError):
        pass

    slots: list[Slot] = []
    for i, value in enumerate(args):
        kind = _kind(value)
        if kind is None:
            continue
        if i < len(positional):
            name = positional[i]
        elif varargs is not None:
            name = f"*{varargs}[{i - len(positional)}]"
        else:
            name = f"arg{i}"
        slots.append(Slot(name, "args", i, kind))
    for key, value in kwargs.items():
        kind = _kind(value)
        if kind is not None:
            slots.append(Slot(key, "kwargs", key, kind))
    return slots


def _kind(value: Any) -> Literal["frame", "series"] | None:
    if isinstance(value, pd.DataFrame):
        return "frame"
    if isinstance(value, pd.Series):
        return "series"
    return None


def snapshot(value: Any) -> Any:
    """Copy an argument so that later mutation by the test can't reach the copy.

    Parameters
    ----------
    value : Any
        The argument.

    Returns
    -------
    Any
        A deep copy, or ``value`` itself when it cannot be copied.
    """
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy(deep=True)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


@dataclass(frozen=True)
class ResultSummary:
    """The shape of a target's return value, enough to notice it changing.

    Attributes
    ----------
    kind : str
        ``"frame"``, ``"series"``, ``"scalar"``, ``"none"`` or ``"other"``.
    numeric : bool
        Whether the value (or every column of it) is numeric.
    all_na : bool
        Whether the value is entirely missing. Never true of an empty value.
    type_name : str
        The value's type, or its dtype for a Series.
    """

    kind: str
    numeric: bool
    all_na: bool
    type_name: str

    @classmethod
    def of(cls, value: Any) -> ResultSummary:
        """Summarise a return value.

        Parameters
        ----------
        value : Any
            What the target returned.

        Returns
        -------
        ResultSummary
            Its summary.
        """
        if value is None:
            return cls("none", False, False, "None")
        if isinstance(value, pd.DataFrame):
            dtypes = list(value.dtypes)
            numeric = bool(dtypes) and all(_is_numeric_dtype(d) for d in dtypes)
            all_na = value.size > 0 and bool(value.isna().all().all())
            return cls("frame", numeric, all_na, "DataFrame")
        if isinstance(value, pd.Series):
            all_na = len(value) > 0 and bool(value.isna().all())
            return cls(
                "series", _is_numeric_dtype(value.dtype), all_na, str(value.dtype)
            )
        if _is_number(value):
            all_na = _is_missing(value)
            return cls("scalar", True, all_na, type(value).__name__)
        if isinstance(value, (str, bytes, bool, np.bool_)) or _is_missing(value):
            return cls("scalar", False, _is_missing(value), type(value).__name__)
        return cls("other", False, False, type(value).__name__)


def _is_numeric_dtype(dtype: Any) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(dtype)
    ) and not pd.api.types.is_bool_dtype(dtype)


def _is_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, complex, np.number))


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


@dataclass
class CallSite:
    """One recorded call into a target that passed a DataFrame or Series.

    Attributes
    ----------
    nodeid : str
        The test that made the call.
    target : str
        The target's display name.
    ordinal : int
        Which outermost target call this was within the test, counting from 0.
    func : callable
        The original, unwrapped target.
    args : tuple
        Snapshot of the positional arguments.
    kwargs : dict
        Snapshot of the keyword arguments.
    slots : list of Slot
        The DataFrame and Series arguments.
    result : ResultSummary or None
        What the call returned, or None if it raised.
    raised : str or None
        The exception type the call raised, if it did.
    """

    nodeid: str
    target: str
    ordinal: int
    func: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    slots: list[Slot]
    result: ResultSummary | None
    raised: str | None = None

    def fresh_args(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return the recorded arguments with every frame and Series copied afresh.

        A replay must not see what an earlier replay did to its inputs in place.

        Returns
        -------
        tuple of (tuple, dict)
            Positional and keyword arguments for one replay.
        """
        args, kwargs = self.args, self.kwargs
        for slot in self.slots:
            args, kwargs = slot.replace(
                args, kwargs, slot.get(args, kwargs).copy(deep=True)
            )
        return args, kwargs

    def schema(self) -> tuple[Any, ...]:
        """Return a key for "the same kind of call": target plus argument dtypes.

        Returns
        -------
        tuple
            Hashable description of the target and its probe-able arguments.
        """
        parts: list[Any] = [self.target]
        for slot in self.slots:
            value = slot.get(self.args, self.kwargs)
            if isinstance(value, pd.DataFrame):
                dtypes = tuple((str(c), str(d)) for c, d in value.dtypes.items())
            else:
                dtypes = ((str(value.name), str(value.dtype)),)
            parts.append((slot.name, dtypes, type(value.index).__name__))
        return tuple(parts)


@dataclass
class Injection:
    """A plan to swap one argument of one call during a test re-run.

    Attributes
    ----------
    ordinal : int
        Which outermost target call to change.
    target : str
        The target that call must be to, or the plan doesn't apply.
    slot : Slot
        Which argument to change.
    mutate : callable
        Maps the live argument to its replacement. May raise ``ValueError`` when the
        live argument doesn't fit (a re-run that built a different frame).
    """

    ordinal: int
    target: str
    slot: Slot
    mutate: Callable[[Any], Any]
    reached: bool = False
    mismatch: str | None = None
    raised: BaseException | None = None


@dataclass
class TestContext:
    """What the wrappers should do during one test's call phase.

    Attributes
    ----------
    nodeid : str
        The running test.
    mode : {"record", "inject"}
        Record call sites, or apply ``injection``.
    injection : Injection, optional
        The swap to make, in ``"inject"`` mode.
    """

    __test__ = False  # not a pytest test class, despite the name

    nodeid: str
    mode: Literal["record", "inject"]
    injection: Injection | None = None
    sites: list[CallSite] = field(default_factory=list)
    calls: int = 0


@contextlib.contextmanager
def activate(ctx: TestContext | None) -> Iterator[TestContext | None]:
    """Make ``ctx`` the active context for the duration of the block.

    Parameters
    ----------
    ctx : TestContext or None
        The context; None deactivates recording and injection.

    Yields
    ------
    TestContext or None
        ``ctx``.
    """
    global _ACTIVE
    previous, _ACTIVE = _ACTIVE, ctx
    try:
        yield ctx
    finally:
        _ACTIVE = previous


def call_target(
    name: str, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Call a wrapped target, recording or injecting as the active context says.

    Parameters
    ----------
    name : str
        The target's display name.
    func : callable
        The original, unwrapped target.
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    Any
        Whatever ``func`` returns.
    """
    ctx = _ACTIVE
    depth = _DEPTH.get()
    token = _DEPTH.set(depth + 1)
    try:
        if ctx is None or depth > 0:
            return func(*args, **kwargs)
        ordinal = ctx.calls
        ctx.calls += 1
        if ctx.mode == "inject":
            return _inject(ctx, name, ordinal, func, args, kwargs)
        return _record(ctx, name, ordinal, func, args, kwargs)
    finally:
        _DEPTH.reset(token)


def _record(
    ctx: TestContext,
    name: str,
    ordinal: int,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    slots = find_slots(func, args, kwargs)
    if not slots:
        return func(*args, **kwargs)
    saved_args = tuple(snapshot(a) for a in args)
    saved_kwargs = {k: snapshot(v) for k, v in kwargs.items()}
    site = CallSite(
        ctx.nodeid, name, ordinal, func, saved_args, saved_kwargs, slots, None
    )
    ctx.sites.append(site)
    try:
        result = func(*args, **kwargs)
    except BaseException as exc:
        site.raised = type(exc).__name__
        raise
    site.result = ResultSummary.of(result)
    return result


def _inject(
    ctx: TestContext,
    name: str,
    ordinal: int,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    plan = ctx.injection
    if plan is None or plan.ordinal != ordinal:
        return func(*args, **kwargs)
    if plan.target != name:
        plan.mismatch = f"call {ordinal} went to {name}, not {plan.target}"
        return func(*args, **kwargs)
    try:
        live = plan.slot.get(args, kwargs)
        args, kwargs = plan.slot.replace(args, kwargs, plan.mutate(live))
    except (LookupError, ValueError, TypeError) as exc:
        plan.mismatch = f"argument didn't fit: {exc}"
        return func(*args, **kwargs)
    plan.reached = True
    try:
        return func(*args, **kwargs)
    except BaseException as exc:
        plan.raised = exc
        raise
