from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def test_paddle_cast(x):
    y = x + 1
    return y.cast("int")


def test_paddle_cast2(x):
    y = x + 1
    return paddle.cast(y, "int")


class TestExecutor(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        self.assert_results(test_paddle_cast, a)
        self.assert_results(test_paddle_cast2, a)


if __name__ == "__main__":
    unittest.main()
