# GET_ITER (new)
# FOR_ITER (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def for_loop(x: int, y: paddle.Tensor):
    for i in [1, 2, 3]:
        y += 1
    return y


class TestExecutor(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        self.assert_results(for_loop, 5, a)


if __name__ == "__main__":
    unittest.main()
