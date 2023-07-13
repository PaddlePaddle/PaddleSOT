# New Supported Instructions:
# BUILD_TUPLE
# BINARY_SUBSCR

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: int, y: paddle.Tensor):
    x = (x, y)
    return x[1] + 1


def foo1(x: int, y: paddle.Tensor):
    z = (x, y, 3, 4)
    return z[0:5:1]


def foo2(x: int, y: paddle.Tensor):
    z = (x, y)
    return z[0]


def tuple_count_int(x: int, y: paddle.Tensor):
    z = (x, x, 2, 1)
    return z.count(x)


def tuple_count_tensor(x: paddle.Tensor, y: tuple[paddle.Tensor]):
    return y.count(x)


def tuple_index_int(x: int, y: paddle.Tensor):
    z = (x, y, x, y, y)
    return z.index(x)


def tuple_index_tensor(x: paddle.Tensor, y: tuple[paddle.Tensor]):
    return y.index(x)


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))
        self.assert_results(foo1, 1, paddle.to_tensor(2))
        self.assert_results(foo2, 1, paddle.to_tensor(2))
        self.assert_results(tuple_count_int, 1, paddle.to_tensor(2))
        self.assert_results(tuple_index_int, 1, paddle.to_tensor(2))
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        self.assert_results(tuple_count_tensor, a, (a, b, a, b))
        self.assert_results(tuple_index_tensor, b, (b, b, b, a))


if __name__ == "__main__":
    unittest.main()
