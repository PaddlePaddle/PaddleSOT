from __future__ import annotations

import paddle
from symbolic_trace import symbolic_trace


def foo(x: paddle.Tensor):
    whilespace = " "
    hello_world = f"Hello{whilespace}World"
    x = x + 1
    return x


symbolic_trace(foo)(paddle.to_tensor(1))

# Instructions:
# LOAD_CONST
# STORE_FAST
# FORMAT_VALUE (new)
# BUILD_STRING (new)
# BINARY_ADD
# STORE_FAST
# RETURN_VALUE

# Variables:
# TensorVariable
# ConstantVariable
