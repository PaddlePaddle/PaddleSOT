# BUILD_SLICE (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from sot.psdb import check_no_breakgraph


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


def tensor_subscript_ellipsis(x: paddle.Tensor, y: paddle.Tensor):
    return x[...] + y[...]


@check_no_breakgraph
def tensor_subscript_tensor(x: paddle.Tensor):
    d0, d1 = paddle.shape(x)
    return x[: d0 // 2, d1 // 2 : d1]


class TestSlice(TestCaseBase):
    def test_simple(self):
        x = list(range(10))
        y = paddle.arange(10)
        self.assert_results_with_side_effects(build_list_slice, x, y)
        self.assert_results_with_side_effects(build_list_slice_with_step, x, y)
        self.assert_results_with_side_effects(build_tuple_slice, x, y)
        self.assert_results_with_side_effects(build_tuple_slice_with_step, x, y)


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


class TestLayerList(TestCaseBase):
    def test_layer_list_slice(self):
        layer = MyLayer()
        x = paddle.randn([5, 10])
        self.assert_results(layer_list_slice, layer, x)


def tensor_slice(x: paddle.Tensor):
    return x[1, 1, 1] + 1


class TestTensorSlice(TestCaseBase):
    def test_tensor_slice(self):
        x = paddle.randn([4, 3, 10])
        self.assert_results(tensor_slice, x)


class TestTensorEllipsis(TestCaseBase):
    def test_tensor_subscript_ellipsis(self):
        x = paddle.rand((10,))
        y = paddle.rand((10, 10))
        self.assert_results(tensor_subscript_ellipsis, x, y)


class TestTensorSubscriptTensor(TestCaseBase):
    def test_tensor_subscript_tensor(self):
        x = paddle.rand((10, 10))
        self.assert_results(tensor_subscript_tensor, x)


class LayerListNet(paddle.nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.layer_list = paddle.nn.LayerList(
            [paddle.nn.Linear(5, 5), paddle.nn.Linear(5, 5)]
        )

    def forward(self, x):
        out = self.layer_list[0](x)
        for layer in self.layer_list[1:]:
            out = layer(out)
        return out


class TestLayerListSlice(TestCaseBase):
    def test_layer_list_slice(self):
        x = paddle.randn([2, 5])
        net = LayerListNet()
        self.assert_results(layer_list_slice, net, x)


if __name__ == "__main__":
    unittest.main()
