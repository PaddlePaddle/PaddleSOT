from __future__ import annotations

import paddle
from symbolic_trace import symbolic_trace


def make_fn(x: paddle.Tensor):
    def fn():
        return 1

    return fn() + x


a = paddle.to_tensor(1)
b = paddle.to_tensor(2)
c = paddle.to_tensor(3)
d = paddle.to_tensor(4)
symbolic_trace(make_fn)(a)

# Instructions:
#
# LOAD_CONST
# MAKE_FUNCTION
# STORE_FAST
# LOAD_FAST
# CALL_FUNCTION
# LOAD_CONST
# RETURN_VALUE
# BINARY_ADD

# Variables:
# ConstantVariable
# NestedUserFunctionVariable (new)
# TensorVariable
