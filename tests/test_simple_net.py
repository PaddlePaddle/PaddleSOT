import unittest

from test_case_base import TestCaseBase

import paddle


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = paddle.nn.Linear(10, 3)
        self.linear2 = paddle.nn.Linear(3, 1)

    def forward(self, x):
        out1 = self.linear1(x)
        out2 = self.linear2(out1)
        return out1 + out2


class TestNet(TestCaseBase):
    def test(self):
        inp = paddle.rand((10,))
        net = SimpleNet()
        self.assert_results(net, inp)


if __name__ == "__main__":
    unittest.main()
