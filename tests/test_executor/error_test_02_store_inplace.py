import paddle
from symbolic_trace import symbolic_trace


def simple(x: int, y: paddle.Tensor):
    x = x + 1
    y = y + 1
    x += y
    return x


symbolic_trace(simple)(1, paddle.to_tensor(2))

# Instructions:
# LOAD_FAST
# BINARY_ADD
# STORE_FAST (new)
# INPLACE_ADD (new)
# RETURN_VALUE

# Variables:
# ConstantVariable
# TensorVariable
