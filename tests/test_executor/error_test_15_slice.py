from __future__ import annotations

import paddle
from symbolic_trace import symbolic_trace


def build_slice(x: paddle.Tensor):
    x[2:4] = -1
    return x


def build_slice_with_step(x: paddle.Tensor):
    x[1:5:2] = -1
    return x


a = paddle.arange(10)
b = paddle.arange(10)
symbolic_trace(build_slice)(a)
symbolic_trace(build_slice_with_step)(b)

# Instructions:
#
# LOAD_CONST
# LOAD_FAST
# BUILD_SLICE (new)
# STORE_SUBSCR (new)
# RETURN_VALUE

# Variables:
# ConstantVariable
# SliceVariable (new)
# TensorVariable
