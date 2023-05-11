# MAKE_FUNCTION
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def make_fn(x: paddle.Tensor):
    def fn():
        return 1

    return fn() + x


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(make_fn, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
