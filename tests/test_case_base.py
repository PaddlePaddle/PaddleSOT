import paddle
from symbolic_trace import symbolic_trace
import unittest
import numpy as np

class TestCaseBase(unittest.TestCase):
    def assert_results(self, func, *inputs):
        np.testing.assert_equal(
            symbolic_trace(func)(*inputs), 
            func(*inputs))
        

