import unittest

from test_case_base import TestCaseBase

import paddle


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def foo_1(x: paddle.Tensor):
    m = x + 1
    y = add(m * 3, m * 2)
    return y


def foo_2(x: paddle.Tensor):
    m = x + 1
    y = sub(m * 3, m * 2)
    return y


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo_1, paddle.to_tensor(2))
        self.assert_results(foo_2, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
