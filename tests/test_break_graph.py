import unittest

from test_case_base import TestCaseBase

import paddle
from symbolic_trace.utils.paddle_api_config import add_break_graph_apis


def ifelse_func(x, y):
    if x > 0:
        y = y + 1
    else:
        y = y + 2
    return y


class TestIfElse(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        self.assert_results(ifelse_func, x, y)


def multi_output(x: paddle.Tensor):
    m = x + 1
    if x > 0:
        return m
    else:
        return 2 * m


class TestExecutor(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor(2)
        self.assert_results(multi_output, x)
        x = paddle.to_tensor(-2)
        self.assert_results(multi_output, x)


def print_break_graph(x, y):
    z = x + y
    print(x, z)
    out = y * z * 2
    return out


class TestPrint(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor(2)
        y = paddle.to_tensor(3)
        self.assert_results(print_break_graph, x, y)


def to_tensor_break_graph(x, y):
    z = x + y
    out = y * paddle.to_tensor(2) * z
    return out


class TestToTensor(TestCaseBase):
    def test_simple(self):
        add_break_graph_apis([paddle.to_tensor])
        x = paddle.to_tensor(2)
        y = paddle.to_tensor(3)
        self.assert_results(to_tensor_break_graph, x, y)


def tensor_numpy(x):
    x = paddle.to_tensor(x)
    x.clear_gradient()
    return x


class TestBreakGraphInResumeFn(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor(2)
        self.assert_results(tensor_numpy, x)


def inner_fn(a, b, c, d):
    return a + b * c - d


def multi_stack_args(a, b, c):
    out = inner_fn(a, b, c, paddle.to_tensor(4))
    return out


class TestMultiStackArgs(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor(3)
        self.assert_results(multi_stack_args, a, b, c)


if __name__ == "__main__":
    unittest.main()
