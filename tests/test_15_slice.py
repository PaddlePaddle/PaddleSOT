# BUILD_SLICE (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def build_slice(x: list, y: paddle.Tensor):
    x[2:4] = [0, 1]
    return x[0] + y


def build_slice_with_step(x: list, y: paddle.Tensor):
    x[1:5:2] = [0, 1]
    return x[0] + y


class TestExecutor(TestCaseBase):
    def test_simple(self):
        x = list(range(10))
        y = paddle.arange(10)
        self.assert_results(build_slice, x, y)
        self.assert_results(build_slice_with_step, x, y)


if __name__ == "__main__":
    unittest.main()
