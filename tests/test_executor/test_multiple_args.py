import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x, y):
    ret = x + y
    return ret


class TestMultipleArgs(TestCaseBase):
    def test_multiple_args(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        self.assert_results(foo, x, y)


if __name__ == "__main__":
    unittest.main()
