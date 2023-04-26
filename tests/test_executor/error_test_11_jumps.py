from __future__ import annotations

import paddle
from symbolic_trace import symbolic_trace


def pop_jump_if_false(x: bool, y: paddle.Tensor):
    if x:
        y += 1
    else:
        y -= 1
    return y


def jump_if_false_or_pop(x: bool, y: paddle.Tensor):
    return x and (y + 1)


def jump_if_true_or_pop(x: bool, y: paddle.Tensor):
    return x or (y + 1)


def pop_jump_if_true(x: bool, y: bool, z: paddle.Tensor):
    return (x or y) and z


def jump_absolute(x: int, y: paddle.Tensor):
    while x > 0:
        y += 1
        x -= 1
    return y


a = paddle.to_tensor(1)
b = paddle.to_tensor(2)
c = paddle.to_tensor(3)
d = paddle.to_tensor(4)
symbolic_trace(pop_jump_if_false)(True, a)
symbolic_trace(jump_if_false_or_pop)(True, a)
symbolic_trace(jump_if_true_or_pop)(False, a)
symbolic_trace(pop_jump_if_true)(True, False, a)
symbolic_trace(jump_absolute)(5, a)

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
