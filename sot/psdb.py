from __future__ import annotations

import builtins
import types
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from typing import TypeVar

    from typing_extensions import ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")

NO_BREAKGRAPH_CODES: set[types.CodeType] = set()
NO_FALLBACK_CODES: set[types.CodeType] = set()


def assert_true(input: bool):
    assert input


def print(*args, **kwargs):
    builtins.print("[Dygraph]", *args, **kwargs)


def breakpoint():
    import paddle

    old = paddle.framework.core.set_eval_frame(None)
    builtins.breakpoint()
    paddle.framework.core.set_eval_frame(old)


def check_no_breakgraph(fn: Callable[P, T]) -> Callable[P, T]:
    NO_BREAKGRAPH_CODES.add(fn.__code__)
    return fn


def breakgraph():
    pass


def check_no_fallback(fn: Callable[P, T]) -> Callable[P, T]:
    NO_FALLBACK_CODES.add(fn.__code__)
    return fn


def fallback():
    pass


def in_sot():
    return False
