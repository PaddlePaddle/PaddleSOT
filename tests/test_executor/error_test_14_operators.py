from __future__ import annotations

import paddle
from symbolic_trace import symbolic_trace


def unary_positive(x: int):
    y = +x
    return y


def unary_negative(x: paddle.Tensor):
    y = -x
    return y


def unary_not(x: paddle.Tensor):
    y = not x
    return y


def unary_invert(x: paddle.Tensor):
    y = ~x
    return y


def binary_power(x: paddle.Tensor, y: paddle.Tensor):
    z = x**y
    return z


def binary_multiply(x: paddle.Tensor, y: paddle.Tensor):
    z = x * y
    return z


def binary_matrix_multiply(x: paddle.Tensor, y: paddle.Tensor):
    z = x @ y
    return z


def binary_floor_divide(x: paddle.Tensor, y: paddle.Tensor):
    z = x // y
    return z


def binary_true_divide(x: paddle.Tensor, y: paddle.Tensor):
    z = x / y
    return z


def binary_modulo(x: paddle.Tensor, y: paddle.Tensor):
    z = x % y
    return z


def binary_add(x: paddle.Tensor, y: paddle.Tensor):
    z = x + y
    return z


def binary_subtract(x: paddle.Tensor, y: paddle.Tensor):
    z = x - y
    return z


def binary_lshift(x: int, y: int):
    z = x << y
    return z


def binary_rshift(x: int, y: int):
    z = x >> y
    return z


def binary_and(x: paddle.Tensor, y: paddle.Tensor):
    z = x & y
    return z


def binary_or(x: paddle.Tensor, y: paddle.Tensor):
    z = x | y
    return z


def binary_xor(x: paddle.Tensor, y: paddle.Tensor):
    z = x ^ y
    return z


def inplace_power(x: paddle.Tensor, y: paddle.Tensor):
    x **= y
    return x


def inplace_multiply(x: paddle.Tensor, y: paddle.Tensor):
    x *= y
    return x


def inplace_matrix_multiply(x: paddle.Tensor, y: paddle.Tensor):
    x @= y
    return x


def inplace_floor_divide(x: paddle.Tensor, y: paddle.Tensor):
    x //= y
    return x


def inplace_true_divide(x: paddle.Tensor, y: paddle.Tensor):
    x /= y
    return x


def inplace_modulo(x: paddle.Tensor, y: paddle.Tensor):
    x %= y
    return x


def inplace_add(x: paddle.Tensor, y: paddle.Tensor):
    x += y
    return x


def inplace_subtract(x: paddle.Tensor, y: paddle.Tensor):
    x -= y
    return x


def inplace_lshift(x: paddle.Tensor, y: int):
    x <<= y
    return x


def inplace_rshift(x: paddle.Tensor, y: int):
    x >>= y
    return x


def inplace_and(x: paddle.Tensor, y: paddle.Tensor):
    x &= y
    return x


def inplace_or(x: paddle.Tensor, y: paddle.Tensor):
    x |= y
    return x


def inplace_xor(x: paddle.Tensor, y: paddle.Tensor):
    x ^= y
    return x


a = paddle.to_tensor(1)
b = paddle.to_tensor(True)
c = paddle.to_tensor(3)
d = paddle.to_tensor(4)
e = paddle.to_tensor([[1, 2], [3, 4], [5, 6]], dtype='float32')
f = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
g = paddle.to_tensor(False)
symbolic_trace(unary_positive)(1)
symbolic_trace(unary_negative)(a)
symbolic_trace(unary_not)(b)
symbolic_trace(unary_invert)(b)

symbolic_trace(binary_power)(c, d)
symbolic_trace(binary_multiply)(c, d)
symbolic_trace(binary_matrix_multiply)(e, f)
symbolic_trace(binary_floor_divide)(c, d)
symbolic_trace(binary_true_divide)(c, d)
symbolic_trace(binary_modulo)(c, d)
symbolic_trace(binary_add)(c, d)
symbolic_trace(binary_subtract)(c, d)
symbolic_trace(binary_lshift)(10, 2)
symbolic_trace(binary_rshift)(10, 1)
symbolic_trace(binary_and)(b, g)
symbolic_trace(binary_or)(b, g)
symbolic_trace(binary_xor)(b, g)

symbolic_trace(inplace_power)(c, d)
symbolic_trace(inplace_multiply)(c, d)
symbolic_trace(inplace_matrix_multiply)(e, f)
symbolic_trace(inplace_floor_divide)(c, d)
symbolic_trace(inplace_true_divide)(c, d)
symbolic_trace(inplace_modulo)(c, d)
symbolic_trace(inplace_add)(c, d)
symbolic_trace(inplace_subtract)(c, d)
symbolic_trace(inplace_lshift)(10, 2)
symbolic_trace(inplace_rshift)(10, 1)
symbolic_trace(inplace_and)(b, g)
symbolic_trace(inplace_or)(b, g)
symbolic_trace(inplace_xor)(b, g)

# Instructions:
#
# ops...

# Variables:
# ConstantVariable
# NestedUserFunctionVariable
# TensorVariable
