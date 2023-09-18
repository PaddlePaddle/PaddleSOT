from __future__ import annotations

import enum
import os
import time
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np

import paddle

from .opcode_translator import eval_frame_callback
from .utils import GraphLogger, StepCounter, cost_model, log_do

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")

# Temporarily set the default log level to 2 to get more information in CI log.
os.environ["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "2")


def symbolic_translate(fn: Callable[P, R], **kwargs) -> Callable[P, R]:
    """
    This function is the entry point of PaddleSOT. It sets eval_frame_callback before input
    function to achieve Opcode-level translation. The translation process depends on the
    simulation execution, in which information will be collected, especially the network
    code. After the simulation execution is completed, the network code will be compiled
    into a static graph Program to improve performance.

    Args:
        fn: The input function.

    Returns:
        Callable, The wrapped function.

    Examples:
        >>> # doctest: +SKIP("Cound not get source code of function foo."")
        >>> import paddle
        >>> import numpy as np
        >>> from sot.translate import symbolic_translate
        >>> def foo(cond: paddle.Tensor, x: paddle.Tensor):
        ...     x += 1
        ...     if cond:
        ...         x += 1
        ...     else:
        ...         x -= 1
        ...     return x
        >>> symbolic_translate_foo = symbolic_translate(foo)
        >>> # For the true branch, the output is 2.
        >>> cond = paddle.to_tensor(True)
        >>> x = paddle.to_tensor(0)
        >>> dygraph_out = foo(cond, x)
        >>> symbolic_translate_out = symbolic_translate_foo(cond, x)
        >>> dygraph_out
        Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
        2)
        >>> symbolic_translate_out
        Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
        2)
        >>> np.testing.assert_allclose(
        ...     dygraph_out.numpy(), symbolic_translate_out.numpy()
        ... )
        >>> # For the false branch, the output is 0.
        >>> cond = paddle.to_tensor(False)
        >>> dygraph_out = foo(cond, x)
        >>> symbolic_translate_out = symbolic_translate_foo(cond, x)
        >>> dygraph_out
        Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
        0)
        >>> symbolic_translate_out
        Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
        0)
        >>> np.testing.assert_allclose(
        ...     dygraph_out.numpy(), symbolic_translate_out.numpy()
        ... )

    """

    class State(enum):
        COLLECT_INFO = 1
        RUN_SOT = 2
        RUN_DYN = 3

    state = State.COLLECT_INFO if cost_model() else State.RUN_SOT
    dynamic_time_records = []
    sot_time_records = []
    avg_dyn_time = 0

    def callback(frame):
        return eval_frame_callback(frame, **kwargs)

    def impl_sot(*args: P.args, **kwargs: P.kwargs) -> R:
        assert hasattr(
            fn, "__code__"
        ), "Target function has not code for simulating."
        StepCounter().step(fn.__code__)
        GraphLogger().clear()
        paddle.framework.core.set_eval_frame(callback)
        try:
            outs = fn(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            paddle.framework.core.set_eval_frame(None)

        log_do(1, lambda: GraphLogger().print_info())
        return outs

    def impl_dynamic(*args: P.args, **kwargs: P.kwargs) -> R:
        outs = fn(*args, **kwargs)
        return outs

    def impl(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal state, dynamic_time_records, sot_time_records, avg_dyn_time
        if state == State.RUN_SOT:
            return impl_sot(*args, **kwargs)
        elif state == State.RUN_DYN:
            return impl_dynamic(*args, **kwargs)
        elif state == State.COLLECT_INFO:
            if dynamic_time_records < 10:
                start_time = time.perf_counter()
                outs = impl_dynamic(*args, **kwargs)
                time_cost = time.perf_counter() - start_time
                dynamic_time_records.append(time_cost)
                if len(dynamic_time_records == 10):
                    avg_dyn_time = np.mean(dynamic_time_records)
            else:
                start_time = time.perf_counter()
                outs = impl_sot(*args, **kwargs)
                time_cost = time.perf_counter() - start_time
                sot_time_records.append(time_cost)
                if len(sot_time_records) > 20:
                    avg_sot_time = np.mean(sot_time_records[-10:])
                    if avg_sot_time < avg_dyn_time:
                        state = State.RUN_SOT
                    # Coefficient of Variation
                    elif (
                        np.std(sot_time_records[-10:]) / avg_sot_time < 0.1
                        or len(sot_time_records) > 50
                    ):
                        state = State.RUN_DYN

            return outs

    return impl
