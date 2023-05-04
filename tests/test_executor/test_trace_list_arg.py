from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: list[paddle.Tensor], y: list[paddle.Tensor]):
    return x[0] + y[0]


class Test(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor([3, 4])
        self.assert_results(foo, [a], [b])
        self.assert_results(foo, [b], [a])  # Cache hit
        self.assert_results(foo, [a], [c])  # Cache miss


if __name__ == "__main__":
    unittest.main()
