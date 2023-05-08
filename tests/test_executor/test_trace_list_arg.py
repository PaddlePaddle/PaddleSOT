from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: list[paddle.Tensor], y: list[paddle.Tensor]):
    return x[0] + y[0]


def bar(x: list[paddle.Tensor], y: int, z: int):
    return x[y + z] + 1


class TestTraceListArg(TestCaseBase):
    def test_foo(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor([3, 4])
        self.assert_results(foo, [a], [b])
        self.assert_results(foo, [b], [a])  # Cache hit
        self.assert_results(foo, [a], [c])  # Cache miss

    def test_bar(self):
        a = [paddle.to_tensor(1), paddle.to_tensor(2), paddle.to_tensor(3)]
        b = [paddle.to_tensor(2), paddle.to_tensor(3), paddle.to_tensor(4)]
        self.assert_results(bar, a, 1, 1)
        self.assert_results(bar, a, 2, 0)  # Cache hit
        self.assert_results(bar, b, 1, 1)  # Cache miss


if __name__ == "__main__":
    unittest.main()
