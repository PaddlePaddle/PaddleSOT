import unittest

from test_case_base import TestCaseBase

import paddle


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


class TestExecutor(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(True)
        c = paddle.to_tensor(3)
        d = paddle.to_tensor(4)
        e = paddle.to_tensor([[1, 2], [3, 4], [5, 6]], dtype='float32')
        f = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        g = paddle.to_tensor(False)

        self.assert_results(unary_positive, 1)
        self.assert_results(unary_negative, a)
        # self.assert_results(unary_not, b)
        self.assert_results(unary_invert, b)

        self.assert_results(binary_power, c, d)
        self.assert_results(binary_multiply, c, d)
        self.assert_results(binary_matrix_multiply, e, f)
        self.assert_results(binary_floor_divide, c, d)
        self.assert_results(binary_true_divide, c, d)
        self.assert_results(binary_modulo, c, d)
        self.assert_results(binary_add, c, d)
        self.assert_results(binary_subtract, c, d)
        self.assert_results(binary_lshift, 10, 2)
        self.assert_results(binary_rshift, 10, 1)
        self.assert_results(binary_and, b, g)
        self.assert_results(binary_or, b, g)
        self.assert_results(binary_xor, b, g)

        self.assert_results(inplace_power, c, d)
        self.assert_results(inplace_multiply, c, d)
        self.assert_results(inplace_matrix_multiply, e, f)
        self.assert_results(inplace_floor_divide, c, d)
        self.assert_results(inplace_true_divide, c, d)
        self.assert_results(inplace_modulo, c, d)
        self.assert_results(inplace_add, c, d)
        self.assert_results(inplace_subtract, c, d)
        self.assert_results(inplace_lshift, 10, 2)
        self.assert_results(inplace_rshift, 10, 1)
        self.assert_results(inplace_and, b, g)
        self.assert_results(inplace_or, b, g)
        self.assert_results(inplace_xor, b, g)


def run_not_eq(x: paddle.Tensor, y: int):
    out = paddle.reshape(x, [1, -1]) != y
    out = out.astype('float32')
    return out


class TestNotEq(TestCaseBase):
    def test_not_eq(self):
        x = paddle.to_tensor([2])
        y = 3
        self.assert_results(run_not_eq, x, y)


if __name__ == "__main__":
    unittest.main()
