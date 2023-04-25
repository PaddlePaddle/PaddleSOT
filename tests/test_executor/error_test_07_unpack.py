from __future__ import annotations

import paddle
from symbolic_trace import symbolic_trace


def foo(x: tuple[int, paddle.Tensor]):
    y, z = x
    return z + 1


symbolic_trace(foo)((1, paddle.to_tensor(2)))

# Instructions:
# LOAD_FAST
# UNPACK_SEQUENCE (new)
# STORE_FAST
# LOAD_CONST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# TupleVariable
# TensorVariable
# ConstantVariable
