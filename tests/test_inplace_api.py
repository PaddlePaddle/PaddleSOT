import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x):
    x[0] = 3.0
    return x[0] + x[1]


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, paddle.to_tensor([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
