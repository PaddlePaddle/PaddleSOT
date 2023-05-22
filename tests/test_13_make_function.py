# MAKE_FUNCTION
# CALL_FUNCTION_KW
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def make_fn(x: paddle.Tensor):
    def fn(a, b=2, c=3, d=4):
        return a + b + c + d

    return fn(1) + fn(2, c=5) + x


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(make_fn, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
