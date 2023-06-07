# GET_ITER (new)
# FOR_ITER (new)

from __future__ import annotations

import sys
import unittest

from test_case_base import TestCaseBase

import paddle
from symbolic_trace import symbolic_trace


def gener():
    yield 1
    yield 2
    yield 3


def for_list_1(x: paddle.Tensor):
    for i in [1, 2, 3]:
        x += i

        if x > 2:
            x += 1
        else:
            x -= 1
    return x


def for_list_2(x: paddle.Tensor):
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


def for_for_fallback(x, it):
    for i in [1, 2, 3]:
        for item in it:
            x += item
    return x


def for_break(x: paddle.Tensor, it):
    for i in [1, 2, 3]:
        x += i
        if i == 2:
            break
    for i in it:
        x += i
        if i == 2:
            break
    return x


def for_continue(x: paddle.Tensor, it):
    for i in [1, 2, 3]:
        if i == 2:
            continue
        x += i

    for i in it:
        if i == 2:
            continue
        x += i
    return x


@unittest.skipIf(
    sys.version_info >= (3, 10),
    "Python 3.10 will raise an error, please fix it later.",
)
class TestExecutor(TestCaseBase):
    def test_list(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_list_1, a)

    def test_list_with_fallback(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_list_2, a)

    def test_dict(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_dict, a)

    def test_fallback(self):
        a = paddle.to_tensor(1)

        sym_output = symbolic_trace(for_iter)(a, gener())
        paddle_output = for_iter(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_for_for_fallback(self):
        a = paddle.to_tensor(1)

        sym_output = symbolic_trace(for_iter)(a, gener())
        paddle_output = for_iter(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_for_break(self):
        a = paddle.to_tensor(1)
        sym_output = symbolic_trace(for_break)(a, gener())
        paddle_output = for_break(a, gener())
        self.assert_nest_match(sym_output, paddle_output)

    def test_for_continue(self):
        a = paddle.to_tensor(1)
        sym_output = symbolic_trace(for_continue)(a, gener())
        paddle_output = for_continue(a, gener())
        self.assert_nest_match(sym_output, paddle_output)


if __name__ == "__main__":
    unittest.main()
