"""Find the library callables to probe, and wrap them for the session.

A pattern names a package, module, class or function by dotted path
(``mylib``, ``mylib.api``, ``mylib.api.Frame.plot``), or globs over those paths
(``mylib.api.*``, ``mylib.*.summarise``). Every public function, and every public method
of a public class, *defined* in a matching module is a target. A function defined in a
private module and re-exported from a public one is matched, and reported, under its
public name.

Wrapping rebinds the target on the module or class that defines it, and then in every
loaded module that holds the same object, so ``from mylib import f`` done before the
session started is covered too. Test modules import after this point and so get the
wrapper anyway. References held anywhere else, such as a registry dict or a
``functools.partial``, are not rebound.
"""

from __future__ import annotations

import fnmatch
import functools
import importlib
import inspect
import pkgutil
import sys
import types
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from . import runtime

__all__ = ["Installer", "Target", "resolve"]

_GLOB_CHARS = set("*?[")
WRAPPED_ATTR = "__pandas_type_widener_original__"


@dataclass(frozen=True)
class Target:
    """One callable to wrap.

    Attributes
    ----------
    name : str
        Display name: the public dotted path where one exists.
    owner : module or type
        The namespace that defines it.
    attr : str
        Its attribute name on ``owner``.
    raw : Any
        The attribute as stored: a function, ``staticmethod`` or ``classmethod``.
    func : callable
        The underlying function.
    """

    name: str
    owner: Any
    attr: str
    raw: Any
    func: Callable[..., Any]


def resolve(patterns: Iterable[str]) -> list[Target]:
    """Expand target patterns to the callables they name.

    Parameters
    ----------
    patterns : iterable of str
        Dotted paths or globs, as described in the module docstring.

    Returns
    -------
    list of Target
        Every matching callable, once each, in a stable order.

    Raises
    ------
    ValueError
        If a pattern's package can't be imported.
    """
    found: dict[int, Target] = {}
    for pattern in patterns:
        root = _import_root(pattern)
        modules = list(_walk(root))
        aliases = _public_aliases(modules)
        for module in modules:
            for target in _candidates(module, aliases):
                if id(target.func) in found:
                    continue
                canonical = f"{module.__name__}.{target.attr}"
                if isinstance(target.owner, type):
                    canonical = (
                        f"{module.__name__}.{target.owner.__qualname__}.{target.attr}"
                    )
                if _matches(pattern, target.name) or _matches(pattern, canonical):
                    found[id(target.func)] = target
    return sorted(found.values(), key=lambda t: t.name)


def _import_root(pattern: str) -> types.ModuleType:
    # The whole top-level package is walked, not just the part the pattern names: a
    # function re-exported from ``mylib.api`` may be defined in ``mylib._impl``.
    top = pattern.split(".", 1)[0]
    if not top or _GLOB_CHARS & set(top):
        raise ValueError(f"target pattern {pattern!r} must start with a package name")
    try:
        return importlib.import_module(top)
    except ImportError as exc:
        raise ValueError(
            f"can't import {top!r} for target pattern {pattern!r}: {exc}"
        ) from exc


def _walk(root: types.ModuleType) -> Iterable[types.ModuleType]:
    yield root
    path = getattr(root, "__path__", None)
    if path is None:
        return
    for info in pkgutil.walk_packages(path, prefix=f"{root.__name__}."):
        try:
            yield importlib.import_module(info.name)
        except Exception as exc:  # a broken optional submodule shouldn't stop the run
            warnings.warn(
                f"pandas-type-widener: skipping {info.name}, which failed to import: {exc!r}",
                stacklevel=2,
            )


def _public_aliases(modules: list[types.ModuleType]) -> dict[int, str]:
    """Map each object's id to its most public dotted name among ``modules``."""
    best: dict[int, tuple[tuple[int, int], str]] = {}
    for module in modules:
        privacy = sum(part.startswith("_") for part in module.__name__.split("."))
        for attr, obj in vars(module).items():
            if attr.startswith("_") or not (
                inspect.isfunction(obj) or inspect.isclass(obj)
            ):
                continue
            rank = (privacy, len(module.__name__))
            if id(obj) not in best or rank < best[id(obj)][0]:
                best[id(obj)] = (rank, f"{module.__name__}.{attr}")
    return {key: name for key, (_, name) in best.items()}


def _candidates(module: types.ModuleType, aliases: dict[int, str]) -> Iterable[Target]:
    for attr, obj in list(vars(module).items()):
        if attr.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj) and _wrappable(obj):
            yield Target(
                aliases.get(id(obj), f"{module.__name__}.{attr}"),
                module,
                attr,
                obj,
                obj,
            )
        elif inspect.isclass(obj):
            cls_name = aliases.get(id(obj), f"{module.__name__}.{obj.__qualname__}")
            for meth_name, raw in list(vars(obj).items()):
                if meth_name.startswith("_"):
                    continue
                func = (
                    raw.__func__
                    if isinstance(raw, (staticmethod, classmethod))
                    else raw
                )
                if inspect.isfunction(func) and _wrappable(func):
                    yield Target(f"{cls_name}.{meth_name}", obj, meth_name, raw, func)


def _wrappable(func: Callable[..., Any]) -> bool:
    return not (
        inspect.iscoroutinefunction(func)
        or inspect.isasyncgenfunction(func)
        or hasattr(func, WRAPPED_ATTR)
    )


def _matches(pattern: str, name: str) -> bool:
    if _GLOB_CHARS & set(pattern):
        return fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(
            name, f"{pattern}.*"
        )
    return name == pattern or name.startswith(f"{pattern}.")


def make_wrapper(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``func`` so its calls go through :func:`runtime.call_target`.

    Parameters
    ----------
    name : str
        The target's display name.
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        The wrapper; ``functools.wraps`` keeps its signature visible.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # numpydoc ignore=GL08
        return runtime.call_target(name, func, args, kwargs)

    setattr(wrapper, WRAPPED_ATTR, func)
    return wrapper


@dataclass
class Installer:
    """Install target wrappers, remembering how to take them out again."""

    _undo: list[tuple[Any, str, Any]] = field(default_factory=list)

    def install(self, targets: Iterable[Target]) -> None:
        """Wrap each target where it's defined and wherever it was imported.

        Parameters
        ----------
        targets : iterable of Target
            What to wrap.
        """
        rebind: dict[int, tuple[Any, Any]] = {}
        for target in targets:
            wrapper = make_wrapper(target.name, target.func)
            raw: Any = wrapper
            if isinstance(target.raw, staticmethod):
                raw = staticmethod(wrapper)
            elif isinstance(target.raw, classmethod):
                raw = classmethod(wrapper)
            self._set(target.owner, target.attr, raw)
            if isinstance(target.owner, types.ModuleType):
                rebind[id(target.func)] = (target.func, wrapper)

        for module in list(sys.modules.values()):
            try:
                namespace = vars(module)
            except TypeError:
                continue
            for attr, value in list(namespace.items()):
                hit = rebind.get(id(value))
                if hit is not None and hit[0] is value:
                    self._set(module, attr, hit[1])

    def _set(self, owner: Any, attr: str, value: Any) -> None:
        original = vars(owner)[attr]
        setattr(owner, attr, value)
        self._undo.append((owner, attr, original))

    def uninstall(self) -> None:
        """Put back everything :meth:`install` replaced, newest first."""
        while self._undo:
            owner, attr, original = self._undo.pop()
            setattr(owner, attr, original)
