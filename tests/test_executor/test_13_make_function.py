# MAKE_FUNCTION
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def make_fn(x: paddle.Tensor):
    def fn(z):
        return z + 1

    return fn(1) + x


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(make_fn, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
