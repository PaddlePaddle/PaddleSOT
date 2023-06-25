# New Supported Instructions:
# BUILD_TUPLE
# BINARY_SUBSCR


import operator
import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: int, y: paddle.Tensor):
    x = (x, y)
    return x[1] + 1


def foo1(x: int, y: paddle.Tensor):
    x = (x, y, 1)
    return operator.getitem(x, slice(0, 2))


def foo2(x: int, y: paddle.Tensor):
    z = (x, y, 3, 4)
    return z[0:5:1]


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))
        self.assert_results(foo1, 1, paddle.to_tensor(2))
        self.assert_results(foo2, 1, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
    # a = foo2(1, paddle.to_tensor(2))
    # print(a)
