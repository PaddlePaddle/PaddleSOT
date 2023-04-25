import paddle
from symbolic_trace import symbolic_trace


def foo(x: int, y: paddle.Tensor):
    x = [x, y]
    return x[1] + 1


symbolic_trace(foo)(1, paddle.to_tensor(2))

# Instructions:
# LOAD_FAST
# BUILD_LIST (new)
# BINARY_SUBSCR
# LOAD_CONST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# ConstantVariable
# TensorVariable
# ListVariable (new)
