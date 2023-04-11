import paddle
from symbolic_trace import symbolic_trace
import unittest
import numpy as np

class TestCaseBase(unittest.TestCase):
    def assert_results(self, func, *inputs):
        sym_output = symbolic_trace(func)(*inputs)
        paddle_output = func(*inputs)
        print("sym_output", sym_output)
        print("paddle_output", paddle_output)
        np.testing.assert_allclose(
            sym_output, 
            paddle_output)
        

