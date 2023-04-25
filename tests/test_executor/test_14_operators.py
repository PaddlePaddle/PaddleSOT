from __future__ import annotations

import dis

import torch


@torch.compile
def unary_positive(x: torch.Tensor):
    y = +x
    return y


@torch.compile
def unary_negative(x: torch.Tensor):
    y = -x
    return y


@torch.compile
def unary_not(x: torch.Tensor):
    y = not x
    return y


@torch.compile
def unary_invert(x: torch.Tensor):
    y = ~x
    return y


@torch.compile
def binary_power(x: torch.Tensor, y: torch.Tensor):
    z = x**y
    return z


@torch.compile
def binary_multiply(x: torch.Tensor, y: torch.Tensor):
    z = x * y
    return z


@torch.compile
def binary_matrix_multiply(x: torch.Tensor, y: torch.Tensor):
    z = x @ y
    return z


@torch.compile
def binary_floor_divide(x: torch.Tensor, y: torch.Tensor):
    z = x // y
    return z


@torch.compile
def binary_true_divide(x: torch.Tensor, y: torch.Tensor):
    z = x / y
    return z


@torch.compile
def binary_modulo(x: torch.Tensor, y: torch.Tensor):
    z = x % y
    return z


@torch.compile
def binary_add(x: torch.Tensor, y: torch.Tensor):
    z = x + y
    return z


@torch.compile
def binary_subtract(x: torch.Tensor, y: torch.Tensor):
    z = x - y
    return z


@torch.compile
def binary_lshift(x: torch.Tensor, y: int):
    z = x << y
    return z


@torch.compile
def binary_rshift(x: torch.Tensor, y: int):
    z = x >> y
    return z


@torch.compile
def binary_and(x: torch.Tensor, y: torch.Tensor):
    z = x & y
    return z


@torch.compile
def binary_or(x: torch.Tensor, y: torch.Tensor):
    z = x | y
    return z


@torch.compile
def binary_xor(x: torch.Tensor, y: torch.Tensor):
    z = x ^ y
    return z


@torch.compile
def inplace_power(x: torch.Tensor, y: torch.Tensor):
    x **= y
    return x


@torch.compile
def inplace_multiply(x: torch.Tensor, y: torch.Tensor):
    x *= y
    return x


@torch.compile
def inplace_matrix_multiply(x: torch.Tensor, y: torch.Tensor):
    x @= y
    return x


@torch.compile
def inplace_floor_divide(x: torch.Tensor, y: torch.Tensor):
    x //= y
    return x


@torch.compile
def inplace_true_divide(x: torch.Tensor, y: torch.Tensor):
    x /= y
    return x


@torch.compile
def inplace_modulo(x: torch.Tensor, y: torch.Tensor):
    x %= y
    return x


@torch.compile
def inplace_add(x: torch.Tensor, y: torch.Tensor):
    x += y
    return x


@torch.compile
def inplace_subtract(x: torch.Tensor, y: torch.Tensor):
    x -= y
    return x


@torch.compile
def inplace_lshift(x: torch.Tensor, y: int):
    x <<= y
    return x


@torch.compile
def inplace_rshift(x: torch.Tensor, y: int):
    x >>= y
    return x


@torch.compile
def inplace_and(x: torch.Tensor, y: torch.Tensor):
    x &= y
    return x


@torch.compile
def inplace_or(x: torch.Tensor, y: torch.Tensor):
    x |= y
    return x


@torch.compile
def inplace_xor(x: torch.Tensor, y: torch.Tensor):
    x ^= y
    return x


a = torch.as_tensor(1)
b = torch.as_tensor(True)
c = torch.as_tensor(3)
d = torch.as_tensor(4)
e = torch.as_tensor([[1, 2], [3, 4], [5, 6]])
f = torch.as_tensor([[1, 2, 3], [4, 5, 6]])
g = torch.as_tensor(False)
unary_positive(a)
unary_negative(a)
unary_not(b)
unary_invert(b)

binary_power(c, d)
binary_multiply(c, d)
binary_matrix_multiply(e, f)
binary_floor_divide(c, d)
binary_true_divide(c, d)
binary_modulo(c, d)
binary_add(c, d)
binary_subtract(c, d)
binary_lshift(a, 2)
binary_rshift(c, 1)
binary_and(b, g)
binary_or(b, g)
binary_xor(b, g)

# Paddle 貌似支持的 inplace 操作不多，单测可以先用 int 等数据代替
inplace_power(c, d)
inplace_multiply(c, d)
inplace_matrix_multiply(e, f)
inplace_floor_divide(c, d)
inplace_true_divide(c, d)
inplace_modulo(c, d)
inplace_add(c, d)
inplace_subtract(c, d)
inplace_lshift(a, 2)
inplace_rshift(c, 1)
inplace_and(b, g)
inplace_or(b, g)
inplace_xor(b, g)

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
