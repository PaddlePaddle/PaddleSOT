import dis

import torch


@torch.compile
def foo(x: int, y: torch.Tensor):
    z = {x: y}
    return z[x] + 1


foo(1, torch.as_tensor(2))

# Instructions:
# LOAD_FAST
# BUILD_MAP (new)
# STORE_FAST
# BINARY_SUBSCR
# LOAD_CONST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# ConstantVariable
# TensorVariable
# ConstDictVariable (new)
