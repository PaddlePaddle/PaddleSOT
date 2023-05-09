# FORMAT_VALUE (new)
# BUILD_STRING (new)
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: paddle.Tensor):
    whilespace = " "
    hello_world = f"Hello{whilespace}World"
    x = x + 1
    return x


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
