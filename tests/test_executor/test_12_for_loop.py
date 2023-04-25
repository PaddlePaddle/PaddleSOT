from __future__ import annotations

import dis

import torch


@torch.compile
def for_loop(x: int, y: torch.Tensor):
    for i in range(x):
        y += 1
    return y


a = torch.as_tensor(1)
b = torch.as_tensor(2)
c = torch.as_tensor(3)
d = torch.as_tensor(4)
for_loop(5, a)

# Instructions:
#
# LOAD_GLOBAL
# LOAD_FAST
# CALL_FUNCTION
# GET_ITER (new)
# FOR_ITER (new)
# STORE_FAST
# LOAD_CONST
# INPLACE_ADD
# JUMP_ABSOLUTE
# RETURN_VALUE

# Variables:
# BuiltinVariable (new)
# ConstantVariable
# RangeVariable (new)
# ListIteratorVariable (new)
# TensorVariable
