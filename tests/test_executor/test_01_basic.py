import paddle
from symbolic_trace import symbolic_trace


def simple(x: int, y: paddle.Tensor):
    return x + y


print(symbolic_trace(simple)(1, paddle.to_tensor(2)))

# Instructions:
# LOAD_FAST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# ConstantVariable
# TensorVariable
