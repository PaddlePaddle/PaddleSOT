import paddle
from symbolic_trace import symbolic_trace


def foo(x: int, y: paddle.Tensor):
    x = (x, y)
    return x[1] + 1


print(symbolic_trace(foo)(1, paddle.to_tensor(2)))

# Instructions:
# LOAD_FAST
# BUILD_TUPLE (new)
# BINARY_SUBSCR (new)
# LOAD_CONST (new)
# BINARY_ADD
# RETURN_VALUE

# Variables:
# ConstantVariable
# TensorVariable
# TupleVariable (new)
