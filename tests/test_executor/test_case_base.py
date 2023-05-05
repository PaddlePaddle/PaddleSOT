import unittest

import numpy as np

from symbolic_trace import symbolic_trace


class TestCaseBase(unittest.TestCase):
    def assert_results(self, func, *inputs):
        sym_output = symbolic_trace(func)(*inputs)
        paddle_output = func(*inputs)
        np.testing.assert_allclose(sym_output, paddle_output)


class DoubleTestCase(unittest.TestCase):
    def assert_results(self, func, *inputs):
        traced_func = symbolic_trace(func)
        sym_output = traced_func(*inputs)
        eager_output = func(*inputs)
        np.testing.assert_allclose(sym_output, eager_output)
        sym_output = traced_func(*inputs)
        eager_output = func(*inputs)
        np.testing.assert_allclose(sym_output, eager_output)
