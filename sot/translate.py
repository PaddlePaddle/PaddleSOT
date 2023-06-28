from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

import paddle

from .opcode_translator import eval_frame_callback
from .utils import GraphLogger, log_do

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")


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

    def callback(frame):
        return eval_frame_callback(frame, **kwargs)

    def impl(*args: P.args, **kwargs: P.kwargs) -> R:
        GraphLogger().clear()
        paddle.fluid.core.set_eval_frame(callback)
        try:
            outs = fn(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            paddle.fluid.core.set_eval_frame(None)

        log_do(1, lambda: GraphLogger().print_info())

        return outs

    return impl
