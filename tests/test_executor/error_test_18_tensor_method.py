import unittest

from test_case_base import TestCaseBase

import paddle


def paddle_method(x: paddle.Tensor):
    y = x + 1
    return y.mean()


class TestExecutor(TestCaseBase):
    def test_simple(self):
        x = paddle.rand((10,))
        self.assert_results(paddle_method, x)
        self.assert_results(paddle_method, x)
        self.assert_results(paddle_method, x)


if __name__ == "__main__":
    unittest.main()
