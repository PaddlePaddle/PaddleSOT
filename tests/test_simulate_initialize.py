import unittest

from test_case_base import TestCaseBase

import paddle
from paddle import nn
from sot import symbolic_translate


class A:
    def __init__(self, vals):
        vals.append(1)


def foo(x, y):
    out = nn.Softmax()(paddle.to_tensor([x, y], dtype="float32"))
    return out


def bar(x):
    a = A(x)
    t = paddle.to_tensor(x)
    return t.mean()


class TestInit(TestCaseBase):
    def test_init_paddle_layer(self):
        self.assert_results(foo, 1, 2)

    def test_init_python_object(self):
        sot_output = symbolic_translate(bar)([1.0, 2.0])
        dyn_output = bar([1.0, 2.0])
        self.assert_nest_match(sot_output, dyn_output)


if __name__ == "__main__":
    unittest.main()
