import unittest

from test_case_base import TestCaseBase

import paddle


def bar(x, y):
    return x + y


def foo(x: paddle.Tensor):
    m = x + 1
    y = bar(m * 3, m * 2)
    return y


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
