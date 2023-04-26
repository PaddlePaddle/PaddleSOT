import paddle
from symbolic_trace import symbolic_trace


def bar(x):
    y = x + 1
    return y


def foo(x: paddle.Tensor):
    y = bar(x)
    return y


symbolic_trace(foo)(paddle.to_tensor(2))

# Instructions:
# LOAD_GLOBAL (new)
# LOAD_FAST
# CALL_FUNCTION (new)
# LOAD_CONST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# UserFunctionVariable (new)
# TensorVariable
# ConstantVariable
