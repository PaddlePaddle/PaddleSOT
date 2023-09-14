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
    def _analyse_code_info(self, code_map):
        return {k.co_name: str(v.state) for k, v in code_map.items()}

    def test_case_1(self):
        CodeStatus().clear()
        net = SimpleNet1()
        inp = paddle.rand((10, 10))
        self.assert_results(run_net, net, inp)
        code_infos = self._analyse_code_info(CodeStatus().code_map)
        states = list(code_infos.values())
        # run_net, forward, loop body, resumed part2 in loop body
        assert len([v for v in states if v == "CodeState.WITH_GRAPH"]) == 4
        # resumed part1 in loop body
        assert (
            len([v for v in states if v == "CodeState.CodeState.WITHOUT_GRAPH"])
            == 1
        )

    def test_case_2(self):
        with strict_mode_guard(0):
            CodeStatus().clear()
            net = SimpleNet2()
            inp = paddle.rand((10, 10))
            self.assert_results(run_net, net, inp)
            code_infos = self._analyse_code_info(CodeStatus().code_map)
            states = list(code_infos.values())
            # no graph found because fallback (paddle api will not enter simulate)
            assert len([v for v in states if v == "CodeState.WITH_GRAPH"]) == 0


if __name__ == "__main__":
    unittest.main()
