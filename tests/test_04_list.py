# New Supported Instructions:
# BUILD_LIST (new)
# BINARY_SUBSCR


import operator
import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: int, y: paddle.Tensor):
    x = [x, y]
    return x[1] + 1


def list_getitem(x: int, y: paddle.Tensor):
    z = [x, y]
    return operator.getitem(z, 1) + 1


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))
        self.assert_results(list_getitem, 1, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
