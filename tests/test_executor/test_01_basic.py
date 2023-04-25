import paddle
from symbolic_trace import symbolic_trace


def foo(x: int, y: paddle.Tensor):
    return x + y


print(symbolic_trace(foo)(1, paddle.to_tensor(2)))

# Instructions:
# LOAD_FAST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# ConstantVariable
# TensorVariable
