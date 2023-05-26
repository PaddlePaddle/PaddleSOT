# GET_ITER (new)
# FOR_ITER (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from symbolic_trace import symbolic_trace


def for_list(x: paddle.Tensor):
    for i in [1, 2, 3]:
        x += i

        if i > 2:
            x += 1
        else:
            x -= 1
    return x


def for_dict(x: paddle.Tensor):
    map = {1: 2, 3: 4}
    for k in map.keys():
        x += k

    for v in map.values():
        x += v

    for k, v in map.items():
        x += k
        x += v

    return x


def for_iter(x, it):
    for item in it:
        x += item
    return x


class TestExecutor(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_list, a)

    def test_dict(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_dict, a)

    def test_fallback(self):
        def gener():
            yield 1
            yield 2

        a = paddle.to_tensor(1)

        sym_output = symbolic_trace(for_iter)(a, gener())
        paddle_output = for_iter(a, gener())
        self.assert_nest_match(sym_output, paddle_output)


if __name__ == "__main__":
    unittest.main()
