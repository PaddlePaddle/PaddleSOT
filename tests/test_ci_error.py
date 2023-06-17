import types
import unittest

import numpy as np


class TestCiError(unittest.TestCase):
    def test_ci_error(self):
        assert isinstance(np.sum, (types.FunctionType))


if __name__ == '__main__':
    unittest.main()
