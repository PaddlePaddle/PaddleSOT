import unittest

from test_case_base import TestCaseBase

import paddle


def tensor_method_call_1(x: paddle.Tensor):
    y = x + 1
    return y.mean()


def tensor_method_call_2(a: paddle.Tensor, b: paddle.Tensor):
    c = a.add(b)
    d = c.multiply(a)
    e = d.subtract(b)
    f = e.divide(a)
    g = f.pow(2) + f.abs().sqrt()
    h = (g.abs() + 1).log() - (g / g.max()).exp()
    i = h.sin() + h.cos()
    return i


class TestTensorMethod(TestCaseBase):
    def test_tensor_method_1(self):
        x = paddle.rand([10])
        y = paddle.rand([2, 4, 6])
        self.assert_results(tensor_method_call_1, x)
        self.assert_results(tensor_method_call_1, y)

    def test_tensor_method_2(self):
        x = paddle.rand([42])
        y = paddle.rand([42])
        self.assert_results(tensor_method_call_2, x, y)


if __name__ == "__main__":
    unittest.main()
