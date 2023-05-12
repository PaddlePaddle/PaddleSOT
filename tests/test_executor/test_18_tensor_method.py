import unittest

from test_case_base import TestCaseBase

import paddle


def paddle_method(x: paddle.Tensor):
    y = x + 1
    return y.mean()


class TestTensorMethod(TestCaseBase):
    def test_tensor_method(self):
        x = paddle.rand([10])
        y = paddle.rand([2, 4, 6])
        self.assert_results(paddle_method, x)
        self.assert_results(paddle_method, y)


# TODO: add more tests

if __name__ == "__main__":
    unittest.main()
