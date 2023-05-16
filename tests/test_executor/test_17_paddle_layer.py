import unittest

from test_case_base import TestCaseBase

import paddle


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = paddle.nn.Linear(10, 1)

    def forward(self, x):
        out1 = self.linear1(x)
        return out1


def net_call(x: paddle.Tensor, net):
    return net(x)


def net_call_passed_by_user(x: paddle.Tensor, net_forward):
    return net_forward(x)


class TestLayer(TestCaseBase):
    def test_layer(self):
        x = paddle.rand((10,))
        net = SimpleNet()
        self.assert_results(net_call, x, net)
        self.assert_results(net_call, x, net.forward)


if __name__ == "__main__":
    unittest.main()
