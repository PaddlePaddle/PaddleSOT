# FORMAT_VALUE (new)
# BUILD_STRING (new)
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from sot.psdb import assert_true


def foo(x: paddle.Tensor):
    whilespace = 123
    hello_world = f"Hello {whilespace} World"
    z = assert_true(hello_world == "Hello 123 World")
    x = x + 1
    return x


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
