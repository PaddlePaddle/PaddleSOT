# BUILD_SLICE (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def build_list_slice(x: list, y: paddle.Tensor):
    x[2:4] = [0, 1]
    return x[0] + y


def build_list_slice_with_step(x: list, y: paddle.Tensor):
    x[1:5:2] = [0, 1]
    return x[0] + y


def build_tuple_slice(x: list, y: paddle.Tensor):
    x[2:4] = (0, 1)
    return x[0] + y


def build_tuple_slice_with_step(x: list, y: paddle.Tensor):
    x[1:5:2] = (0, 1)
    return x[0] + y


class TestExecutor(TestCaseBase):
    def test_simple(self):
        x = list(range(10))
        y = paddle.arange(10)
        self.assert_results(build_list_slice, x, y)
        self.assert_results(build_list_slice_with_step, x, y)
        self.assert_results(build_tuple_slice, x, y)
        self.assert_results(build_tuple_slice_with_step, x, y)


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linears = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for i in range(10)]
        )

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


def layer_list_slice(layer, x):
    out = layer(x)
    return out


# TODO(zmh): support instance
class TestLayerList(TestCaseBase):
    def test_run(self):
        layer = MyLayer()
        x = paddle.randn([5, 10])
        # self.assert_results(layer_list_slice, layer, x)


if __name__ == "__main__":
    unittest.main()
