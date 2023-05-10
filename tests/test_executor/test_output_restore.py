from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: paddle.Tensor, y: paddle.Tensor, z: int):
    a = x + 1
    b = z + 1
    l = [1, a, b, y]
    return l


class TestOutputRestore(TestCaseBase):
    def test_foo(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)

        self.assert_results(foo, a, b, 3)


if __name__ == "__main__":
    unittest.main()
