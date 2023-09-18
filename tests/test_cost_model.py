import time
import unittest

from test_case_base import TestCaseBase, cost_model_guard

import paddle
from sot import psdb, symbolic_translate


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


class TestCostModel(TestCaseBase):
    @cost_model_guard("True")
    def test_dyn_fast(self):
        x = paddle.rand([10])
        net = paddle.nn.Linear(10, 10)
        sot_fn = symbolic_translate(dyn_fast)
        for i in range(30):
            sot_fn(x, net, iter(range(10)))

        State_idx = sot_fn.__code__.co_freevars.index("State")
        State = sot_fn.__closure__[State_idx].cell_contents

        state_idx = sot_fn.__code__.co_freevars.index("state")
        state = sot_fn.__closure__[state_idx].cell_contents

        assert state == State.RUN_DYN

    @cost_model_guard("True")
    def test_sot_fast_with_multi_graph(self):
        x = paddle.rand([10])
        net = paddle.nn.Linear(10, 10)
        sot_fn = symbolic_translate(sot_fast_with_multi_graph)
        for i in range(30):
            sot_fn(x, net)

        State_idx = sot_fn.__code__.co_freevars.index("State")
        State = sot_fn.__closure__[State_idx].cell_contents

        state_idx = sot_fn.__code__.co_freevars.index("state")
        state = sot_fn.__closure__[state_idx].cell_contents

        assert state == State.RUN_SOT

    @cost_model_guard("True")
    def test_sot_fast_with_single_graph(self):
        x = paddle.rand([10])
        net = paddle.nn.Linear(10, 10)
        sot_fn = symbolic_translate(sot_fast_with_single_graph)
        for i in range(30):
            sot_fn(x, net)

        State_idx = sot_fn.__code__.co_freevars.index("State")
        State = sot_fn.__closure__[State_idx].cell_contents

        state_idx = sot_fn.__code__.co_freevars.index("state")
        state = sot_fn.__closure__[state_idx].cell_contents

        assert state == State.RUN_SOT


if __name__ == "__main__":
    unittest.main()
