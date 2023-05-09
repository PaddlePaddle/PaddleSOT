from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def rot_two_return_a(a: paddle.Tensor, b: paddle.Tensor):
    b, a = a, b
    return a + 1


def rot_two_return_b(a: paddle.Tensor, b: paddle.Tensor):
    b, a = a, b
    return b + 2


def rot_three_return_a(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor):
    a, b, c = c, b, a
    return a + 1


def rot_three_return_b(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor):
    a, b, c = c, b, a
    return b + 1


def rot_three_return_c(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor):
    a, b, c = c, b, a
    return c + 1


def rot_four_return_a(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    a, b, c, d = d, c, b, a
    return a + 1


def rot_four_return_b(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    a, b, c, d = d, c, b, a
    return b + 1


def rot_four_return_c(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    a, b, c, d = d, c, b, a
    return c + 1


def rot_four_return_d(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    a, b, c, d = d, c, b, a
    return d + 1


class TestExecutor(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor(3)
        d = paddle.to_tensor(4)
        self.assert_results(rot_two_return_a, a, b)
        self.assert_results(rot_two_return_b, a, b)

        self.assert_results(rot_three_return_a, a, b, c)
        self.assert_results(rot_three_return_b, a, b, c)
        self.assert_results(rot_three_return_c, a, b, c)

        self.assert_results(rot_four_return_a, a, b, c, d)
        self.assert_results(rot_four_return_b, a, b, c, d)
        self.assert_results(rot_four_return_c, a, b, c, d)
        self.assert_results(rot_four_return_d, a, b, c, d)


if __name__ == "__main__":
    unittest.main()
