import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: int, y: paddle.Tensor):
    x = x + 1
    y = y + 1
    x += y
    return x


class TestStoreInplace(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()


# Instructions:
# LOAD_FAST
# BINARY_ADD
# STORE_FAST (new)
# INPLACE_ADD (new)
# RETURN_VALUE

# Variables:
# ConstantVariable
# TensorVariable
