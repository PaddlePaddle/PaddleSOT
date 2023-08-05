from __future__ import annotations

import math
import operator
import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle


def dispatch_len(x: paddle.Tensor):
    return len(x.shape)


def dispatch_tensor_len(x: paddle.Tensor):
    return len(x)


def dispatch_bool(x: paddle.Tensor):
    return operator.truth(x.shape) and bool(x.shape)


def dispatch_ceil(x: paddle.Tensor | float):
    return math.ceil(x) + 1


def dispatch_floor(x: paddle.Tensor | float):
    return math.floor(x) + 1


class TestBuiltinDispatch(TestCaseBase):
    def test_dispatch_len(self):
        self.assert_results(dispatch_len, paddle.to_tensor([1, 2, 3]))

    def test_dispatch_bool(self):
        self.assert_results(dispatch_bool, paddle.to_tensor([1, 2, 3]))

    def test_dispatch_tensor_len(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dispatch_tensor_len, paddle.to_tensor([1, 2, 3])
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                dispatch_tensor_len, paddle.to_tensor([4, 5, 6])
            )
            self.assertEqual(ctx.translate_count, 1)

    def test_not_dispatch_tensor_ceil(self):
        self.assert_results(dispatch_ceil, paddle.to_tensor(1.2))

    def test_dispatch_float_ceil(self):
        self.assert_results(dispatch_ceil, 1.2)

    def test_not_dispatch_tensor_floor(self):
        self.assert_results(dispatch_floor, paddle.to_tensor(1.2))

    def test_dispatch_float_floor(self):
        self.assert_results(dispatch_floor, 1.2)


def run_getattr(x: paddle.Tensor):
    attr = 'dtype'
    out = getattr(x, attr)
    return out


class TestGetattr(TestCaseBase):
    def test_getattr(self):
        x = paddle.to_tensor(4)
        self.assert_results(run_getattr, x)


if __name__ == "__main__":
    unittest.main()
