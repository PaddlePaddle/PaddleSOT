from __future__ import annotations

import dis

import torch


@torch.compile
def make_fn(x: torch.Tensor):
    def fn():
        return 1

    return fn() + x


a = torch.as_tensor(1)
b = torch.as_tensor(2)
c = torch.as_tensor(3)
d = torch.as_tensor(4)
make_fn(a)

# Instructions:
#
# LOAD_CONST
# MAKE_FUNCTION
# STORE_FAST
# LOAD_FAST
# CALL_FUNCTION
# LOAD_CONST
# RETURN_VALUE
# BINARY_ADD

# Variables:
# ConstantVariable
# NestedUserFunctionVariable (new)
# TensorVariable
