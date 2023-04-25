import dis

import torch


def bar(x):
    y = x + 1
    return y


@torch.compile
def foo(x: torch.Tensor):
    y = bar(x)
    return y


foo(torch.as_tensor(2))

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
