from __future__ import annotations

import paddle
from symbolic_trace import symbolic_trace


def foo(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    b, a = a, b
    a, b, c = c, b, a
    # 没有使用 ROT_FOUR，查了一下好像现在基本没有了，
    # Python/compile.c 里的编译产生 ROT_FOUR 条件貌似也很苛刻……
    d, c, b, a = a, b, c, d
    return (a + 1, b + 2, c + 3, d + 4)


symbolic_trace(foo)(
    paddle.to_tensor(1),
    paddle.to_tensor(2),
    paddle.to_tensor(3),
    paddle.to_tensor(4),
)

# Instructions:
# LOAD_FAST
# ROT_TWO (new)
# STORE_FAST
# ROT_THREE (new)
# BUILD_TUPLE
# UNPACK_SEQUENCE
# LOAD_CONST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# TupleVariable
# TensorVariable
# ConstantVariable
