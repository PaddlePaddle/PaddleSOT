from __future__ import annotations

import dis

import torch


@torch.compile
def foo(x: torch.Tensor):
    whilespace = " "
    hello_world = f"Hello{whilespace}World"
    x = x + 1
    return x


foo(torch.as_tensor(1))

# Instructions:
# LOAD_CONST
# STORE_FAST
# FORMAT_VALUE (new)
# BUILD_STRING (new)
# BINARY_ADD
# STORE_FAST
# RETURN_VALUE

# Variables:
# TensorVariable
# ConstantVariable
