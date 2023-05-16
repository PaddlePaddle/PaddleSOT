from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def func_dup_top_1():
    return True == True != False


def func_dup_top_2(x):
    y = x + 1
    return True == True != False


def func_dup_top_two(x: list[paddle.Tensor]):
    x[0] += x[1]
    return x


class TestDupTop(TestCaseBase):
    def test_dup_top(self):
        self.assert_results(func_dup_top_1)
        self.assert_results(func_dup_top_2, paddle.to_tensor(1.0))
        # TODO: fix this after we support side effect
        # self.assert_results(
        #     func_dup_top_two, [paddle.to_tensor(1.0), paddle.to_tensor(2.0)]
        # )


if __name__ == "__main__":
    unittest.main()
