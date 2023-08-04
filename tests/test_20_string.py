from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from sot.utils import ASSERT


def string_format(x: paddle.Tensor):
    whilespace = 123
    hello_world = f"Hello {whilespace} World"
    z = ASSERT(hello_world == "Hello 123 World")
    hello_world2 = f"Hello {whilespace}{whilespace} World"
    z = ASSERT(hello_world2 == "Hello 123123 World")
    hello_world_lower = "Hello World".lower()
    z = ASSERT(hello_world_lower == "hello world")
    return x + 1


def string_lower(x: paddle.Tensor):
    hello_world_lower = "Hello World".lower()
    z = ASSERT(hello_world_lower == "hello world")
    return x + 1


class TestExecutor(TestCaseBase):
    def test_string_format(self):
        self.assert_results(string_format, paddle.to_tensor(1))

    def test_string_lower(self):
        self.assert_results(string_lower, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
