from __future__ import annotations

import dis

import torch


@torch.compile
def foo(x: tuple[int, torch.Tensor]):
    y, z = x
    return z + 1


foo((1, torch.as_tensor(2)))

# Instructions:
# LOAD_FAST
# UNPACK_SEQUENCE (new)
# STORE_FAST
# LOAD_CONST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# TupleVariable
# TensorVariable
# ConstantVariable
