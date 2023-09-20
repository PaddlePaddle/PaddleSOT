import time
import unittest

from test_case_base import TestCaseBase, cost_model_guard

import paddle
from sot import psdb, symbolic_translate
from sot.utils import StepInfoManager, StepState


def dyn_fast(x, net, iter_):
    for i in iter_:
        x = net(x)
    return x


def sot_fast_with_single_graph(x, net):
    if not psdb.in_sot():
        time.sleep(0.1)
    return x + 1


def sot_fast_with_multi_graph(x, net):
    if not psdb.in_sot():
        time.sleep(0.1)
    x = x + 1
    psdb.breakgraph()
    x = x + 2
    return x


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 10)

    def forward(self, x):
        if not psdb.in_sot():
            time.sleep(0.1)
        x = x / 3
        x = x + 5
        x = self.linear(x)
        return x


class TestCostModel(TestCaseBase):
    @cost_model_guard("True")
    def test_dyn_fast(self):
        x = paddle.rand([10])
        net = paddle.nn.Linear(10, 10)
        sot_fn = symbolic_translate(dyn_fast)
        for i in range(60):
            sot_fn(x, net, iter(range(10)))

        state = StepInfoManager().current_state
        assert state == StepState.RUN_DYN

    @cost_model_guard("True")
    def test_sot_fast_with_multi_graph(self):
        x = paddle.rand([10])
        net = paddle.nn.Linear(10, 10)
        sot_fn = symbolic_translate(sot_fast_with_multi_graph)
        for i in range(30):
            sot_fn(x, net)

        state = StepInfoManager().current_state
        assert state == StepState.RUN_SOT

    @cost_model_guard("True")
    def test_sot_fast_with_single_graph(self):
        x = paddle.rand([10])
        net = paddle.nn.Linear(10, 10)
        for i in range(30):
            symbolic_translate(sot_fast_with_single_graph)(x, net)

        state = StepInfoManager().current_state
        assert state == StepState.RUN_SOT

    @cost_model_guard("True")
    def test_net(self):
        x = paddle.rand([10])
        net = Net()
        net = paddle.jit.to_static(net, enable_fallback=True)
        for i in range(30):
            x = net(x)

        state = StepInfoManager().current_state
        assert state == StepState.RUN_SOT


if __name__ == "__main__":
    unittest.main()
