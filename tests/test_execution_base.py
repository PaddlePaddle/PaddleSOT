import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.fluid.compiler import BuildStrategy
from sot import symbolic_translate


def func(x, y):
    ret = 2 * x
    ret = paddle.nn.functional.relu(ret)
    ret = ret + y
    return ret


def simple(x):
    ret = 2 * x
    return ret


class TestExecutor(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        self.assert_results(simple, x)
        self.assert_results(simple, y)


def foo(x):
    out = x + 1
    out = out * 2
    out = paddle.nn.functional.relu(out)
    return out


class TestBackend(TestCaseBase):
    def test_backend(self):
        x = paddle.randn([2, 3])
        dy_out = foo(x)
        sot_out = symbolic_translate(
            foo, build_strategy=BuildStrategy(), backend='CINN'
        )(x)
        self.assert_nest_match(dy_out, sot_out)


if __name__ == "__main__":
    unittest.main()
