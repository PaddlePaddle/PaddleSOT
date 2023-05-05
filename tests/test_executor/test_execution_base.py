import unittest

from test_case_base import TestCaseBase, DoubleTestCase

import paddle


def func(x, y):
    ret = 2 * x
    ret = paddle.nn.functional.relu(ret)
    ret = ret + y
    return ret


def simple(x):
    ret = 2 * x
    return ret


class TestExecutor(DoubleTestCase):
    def test_simple(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        self.assert_results(simple, x)
        #self.assert_results(simple, y)


if __name__ == "__main__":
    unittest.main()
