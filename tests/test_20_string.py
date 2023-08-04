# FORMAT_VALUE (new)
# BUILD_STRING (new)
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from sot.utils import ASSERT


def foo(x: paddle.Tensor):
    whilespace = 123
    hello_world = f"Hello {whilespace} World"
    z = ASSERT(hello_world == "Hello 123 World")
    hello_world2 = f"Hello {whilespace}{whilespace} World"
    z = ASSERT(hello_world2 == "Hello 123123 World")
    hello_world_lower = "Hello World".lower()
    z = ASSERT(hello_world_lower == "hello world")
    return x


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
