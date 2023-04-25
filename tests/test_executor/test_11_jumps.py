from __future__ import annotations

import dis

import torch


@torch.compile
def pop_jump_if_false(x: bool, y: torch.Tensor):
    if x:
        y += 1
    else:
        y -= 1
    return y


@torch.compile
def jump_if_false_or_pop(x: bool, y: torch.Tensor):
    return x and (y + 1)


@torch.compile
def jump_if_true_or_pop(x: bool, y: torch.Tensor):
    return x or (y + 1)


@torch.compile
def pop_jump_if_true(x: bool, y: bool, z: torch.Tensor):
    return (x or y) and z


@torch.compile
def jump_absolute(x: int, y: torch.Tensor):
    while x > 0:
        y += 1
        x -= 1
    return y


a = torch.as_tensor(1)
b = torch.as_tensor(2)
c = torch.as_tensor(3)
d = torch.as_tensor(4)
pop_jump_if_false(True, a)
jump_if_false_or_pop(True, a)
jump_if_true_or_pop(False, a)
pop_jump_if_true(True, False, a)
jump_absolute(5, a)

# TODO: JUMP_FORWARD

# Instructions:
# POP_JUMP_IF_FALSE (new)
# LOAD_CONST
# INPLACE_ADD
# STORE_FAST
# RETURN_VALUE
# INPLACE_SUBTRACT (new)
# JUMP_IF_FALSE_OR_POP (new)
# BINARY_ADD
# JUMP_IF_TRUE_OR_POP (new)
# POP_JUMP_IF_TRUE (new)
# COMPARE_OP (new)

# Variables:
# ConstantVariable
# TensorVariable
