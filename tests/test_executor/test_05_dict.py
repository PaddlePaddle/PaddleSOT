# New Supported Instructions:
# BUILD_MAP (new)

import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: int, y: paddle.Tensor):
    z = {x: y}
    return z[x] + 1


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
