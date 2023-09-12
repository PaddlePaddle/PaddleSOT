import unittest

from test_case_base import TestCaseBase, strict_mode_guard

import paddle
import sot
from sot.utils import CodeStatus


class SimpleNet1(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layers = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for _ in range(30)]
        )

    def forward(self, x):
        for i in range(len(self.layers)):
            sot.psdb.breakgraph()
            x = self.layers[i](x)
            x = self.layers[i](x)
            x = self.layers[i](x)
            x = self.layers[i](x)
        return x


class SimpleNet2(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layers = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for _ in range(30)]
        )

    def forward(self, x):
        sot.psdb.fallback()
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.layers[i](x)
            x = self.layers[i](x)
            x = self.layers[i](x)
        return x


def run_net(net, x):
    for i in range(20):
        x = net(x)
    return x


class TestCodeInfo(TestCaseBase):
    def test_case_1(self):
        CodeStatus().clear()
        net = SimpleNet1()
        inp = paddle.rand((10, 10))
        self.assert_results(run_net, net, inp)
        assert len(CodeStatus().code_map) > 3
        assert CodeStatus().skip_count == 2

    def test_case_2(self):
        with strict_mode_guard(0):
            CodeStatus().clear()
            net = SimpleNet2()
            inp = paddle.rand((10, 10))
            self.assert_results(run_net, net, inp)
            assert len(CodeStatus().code_map) == 3
            assert CodeStatus().skip_count == 3


if __name__ == "__main__":
    unittest.main()
