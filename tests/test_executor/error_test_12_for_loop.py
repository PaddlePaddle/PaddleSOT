from __future__ import annotations

import paddle
from symbolic_trace import symbolic_trace


def for_loop(x: int, y: paddle.Tensor):
    for i in range(x):
        y += 1
    return y


a = paddle.to_tensor(1)
b = paddle.to_tensor(2)
c = paddle.to_tensor(3)
d = paddle.to_tensor(4)
symbolic_trace(for_loop)(5, a)

# Instructions:
#
# LOAD_GLOBAL
# LOAD_FAST
# CALL_FUNCTION
# GET_ITER (new)
# FOR_ITER (new)
# STORE_FAST
# LOAD_CONST
# INPLACE_ADD
# JUMP_ABSOLUTE
# RETURN_VALUE

# Variables:
# BuiltinVariable (new)
# ConstantVariable
# RangeVariable (new)
# ListIteratorVariable (new)
# TensorVariable
