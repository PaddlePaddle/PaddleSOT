import unittest

import numpy as np
from test_case_base import TestCaseBase, strict_mode_guard

import paddle


def foo(x, y):
    ret = x + y
    return ret


class TestNumpy(TestCaseBase):
    def test_tensor_add_numpy_number(self):
        x = paddle.to_tensor([1.0])
        y = np.int64(2)
        self.assert_results(foo, x, y)
        self.assert_results(foo, y, x)

    @strict_mode_guard(0)
    def test_tensor_add_numpy_array(self):
        x = paddle.to_tensor([1.0])
        y = np.array(2.0)
        self.assert_results(foo, x, y)
        self.assert_results(foo, y, x)


if __name__ == "__main__":
    unittest.main()
