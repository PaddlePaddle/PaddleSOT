import paddle
from symbolic_trace import symbolic_trace
import unittest
from test_case_base import TestCaseBase

import paddle
import numpy as np
import random
from numpy.testing import assert_array_equal

def func(x, y):
    return x * y

class TestNet(TestCaseBase):
    def test(self):
        x = paddle.to_tensor([5, 3])
        y = paddle.to_tensor([1, 3])
        x.stop_gradient = False
        self.assert_results(func, x, y)

        ret = symbolic_trace(func)(x, y)
        ret.backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 3.0])

if __name__ == "__main__":
    unittest.main()

