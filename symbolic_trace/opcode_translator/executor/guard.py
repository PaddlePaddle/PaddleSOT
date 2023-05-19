from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable

from ...utils import log

Guard = Callable[[types.FrameType], bool]


@dataclass
class StringifyExpression:
    expr: str
    free_vars: dict[str, Any]


def union_free_vars(*free_vars: dict[str, Any]):
    return {k: v for d in free_vars for k, v in d.items()}


def make_guard(stringify_guards: list[StringifyExpression]) -> Guard:
    free_vars = union_free_vars(
        *[guard.free_vars for guard in stringify_guards]
    )
    num_guards = len(stringify_guards)
    if not num_guards:
        return lambda frame: True
    guard_string = f"lambda frame: {' and '.join([guard.expr for guard in stringify_guards])}"
    guard = eval(
        guard_string,
        free_vars,
    )
    log(3, f"[Guard]: {guard_string}\n")
    assert callable(guard), "guard must be callable."

    return guard
