import unittest

from test_case_base import TestCaseBase

import paddle
from sot import symbolic_translate


def simple(x, y):
    x[0] = 3.0
    z = [y]
    y[1] = 5.0
    return x[0] + x[1] + z[0][1] + y[0] + y[1]


def inplace_in_if(x, y, z):
    if z:
        x[0] = 3.0
        z = [y]
        y[1] = 5.0
        ret = x[0] + x[1] + z[0][1] + y[0] + y[1]
    else:
        return None


def inplace_in_if_fallback(x, y, z):
    if z > 0:
        x[0] = 3.0
        z = [y]
        y[1] = 5.0
        ret = x[0] + x[1] + z[0][1] + y[0] + y[1]
    else:
        return None


def inplace_in_loop(x, y):
    ret = 0
    for i in range(10):
        x[0] = 1
        z = [y]
        y[1] = 2 * i + 1
        ret += x[0] + x[1] + z[0][1] + y[0] + y[1]
    return ret


def inplace_in_loop_fallback(x, y, it):
    ret = 0
    for i in it:
        x[0] = 1
        z = [y]
        y[1] = 2 * i + 1
        ret += x[0] + x[1] + z[0][1] + y[0] + y[1]
    return ret


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(
            simple, paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 4.0])
        )

    def test_if(self):
        self.assert_results(
            inplace_in_if,
            paddle.to_tensor([1.0, 2.0]),
            paddle.to_tensor([3.0, 4.0]),
            True,
        )
        self.assert_results(
            inplace_in_if_fallback,
            paddle.to_tensor([1.0, 2.0]),
            paddle.to_tensor([3.0, 4.0]),
            paddle.to_tensor(1),
        )

    def test_loop(self):
        self.assert_results(
            inplace_in_loop,
            paddle.to_tensor([1.0, 2.0]),
            paddle.to_tensor([3.0, 4.0]),
        )

        a = range(10)
        sym_output = symbolic_translate(inplace_in_loop_fallback)(
            paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 4.0]), iter(a)
        )
        paddle_output = inplace_in_loop_fallback(
            paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 4.0]), iter(a)
        )
        self.assert_nest_match(sym_output, paddle_output)


if __name__ == "__main__":
    unittest.main()
