# New Supported Instructions:
# UNPACK_SEQUENCE (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: tuple[int, paddle.Tensor]):
    y, z = x
    return z + 1


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, (1, paddle.to_tensor(2)))


if __name__ == "__main__":
    unittest.main()
