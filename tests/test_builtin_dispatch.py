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


def dispatch_tensor_reversed(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return reversed([x, y, x, y])


def dispatch_bool(x: paddle.Tensor):
    return operator.truth(x.shape) and bool(x.shape)


def dispatch_ceil(x: paddle.Tensor | float):
    return math.ceil(x) + 1


def dispatch_floor(x: paddle.Tensor | float):
    return math.floor(x) + 1


def test_sum_tuple(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return sum((x, y), x)


def test_sum_list(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return sum([x, y], x)


def test_tensor_sum(x: paddle.Tensor):
    return sum(x)


def test_tensor_sum_api(x: paddle.Tensor):
    return x.sum()


def test_pow(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return pow(x, y, x)


def test_tensor_pow_api(x: paddle.Tensor, y: paddle.Tensor | int):
    return x.pow(y)


def test_math_pow(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return math.pow(x, y)


def test_chr(x: int | hex | paddle.Tensor):
    return chr(x)


def test_ord(x: str):
    return ord(x)


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

    def test_dispatch_tensor_reversed(self):
        self.assert_results_with_side_effects(
            dispatch_tensor_reversed, paddle.to_tensor(1), 2
        )
        self.assert_results_with_side_effects(
            dispatch_tensor_reversed, 2, paddle.to_tensor(1)
        )

    def test_not_dispatch_tensor_ceil(self):
        # ceil should break graph, since it returns a int rather than a tensor
        self.assert_results(dispatch_ceil, paddle.to_tensor(1.2))

    def test_dispatch_float_ceil(self):
        self.assert_results(dispatch_ceil, 1.2)

    def test_not_dispatch_tensor_floor(self):
        # floor should break graph, since it returns a int rather than a tensor
        self.assert_results(dispatch_floor, paddle.to_tensor(1.2))

    def test_dispatch_float_floor(self):
        self.assert_results(dispatch_floor, 1.2)

    def test_dispatch_sum(self):
        self.assert_results(test_sum_tuple, 1, 1)
        self.assert_results(test_sum_tuple, paddle.to_tensor(1), 1)
        self.assert_results(
            test_sum_tuple, paddle.to_tensor(1), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_tuple, paddle.to_tensor([1, 2]), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_tuple, paddle.to_tensor([1, 2]), paddle.to_tensor([1, 3])
        )
        self.assert_results(test_sum_list, 1, 1)
        self.assert_results(test_sum_list, paddle.to_tensor(1), 1)
        self.assert_results(
            test_sum_list, paddle.to_tensor(1), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_list, paddle.to_tensor([1, 2]), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_list, paddle.to_tensor([1, 2]), paddle.to_tensor([1, 3])
        )
        self.assert_results(test_tensor_sum, paddle.to_tensor([1, 2]))
        self.assert_results(test_tensor_sum, paddle.to_tensor((1, 2)))
        self.assert_results(test_tensor_sum_api, paddle.to_tensor([1, 2]))
        self.assert_results(test_tensor_sum_api, paddle.to_tensor((1, 2)))

    def test_dispatch_pow(self):
        self.assert_results(test_pow, 2, 3)
        self.assert_results(test_pow, paddle.to_tensor(2), 3)
        self.assert_results(test_pow, paddle.to_tensor(2), paddle.to_tensor(3))
        self.assert_results(test_math_pow, 2, 3)
        self.assert_results(test_math_pow, paddle.to_tensor(2), 3)
        self.assert_results(
            test_math_pow, paddle.to_tensor(2), paddle.to_tensor(3)
        )
        self.assert_results(test_tensor_pow_api, paddle.to_tensor(2), 3)
        self.assert_results(
            test_tensor_pow_api, paddle.to_tensor(2), paddle.to_tensor(3)
        )

    def test_dispatch_chr(self):
        self.assert_results(test_chr, 65)
        self.assert_results(test_chr, 0x41)
        self.assert_results(test_chr, paddle.to_tensor(65))
        self.assert_results(test_chr, paddle.to_tensor(0x41))

    def test_dispatch_ord(self):
        self.assert_results(test_ord, "a")


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
