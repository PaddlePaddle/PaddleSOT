import dis

import torch


@torch.compile
def foo(x: int, y: torch.Tensor):
    x = [x, y]
    return x[1] + 1


foo(1, torch.as_tensor(2))

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
