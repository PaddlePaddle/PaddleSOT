import unittest

from test_case_base import TestCaseBase, strict_mode_guard

import paddle


def foo(x: int, y: paddle.Tensor):
    return x + y


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))


def numpy_add(x, y):
    out = paddle.to_tensor(x.numpy() + y.numpy())
    return out


class TestNumpyAdd(TestCaseBase):
    @strict_mode_guard(0)
    def test_numpy_add(self):
        x = paddle.to_tensor([2])
        y = paddle.to_tensor([3])
        self.assert_results(numpy_add, x, y)


if __name__ == "__main__":
    unittest.main()


# Instructions:
# LOAD_FAST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# ConstantVariable
# TensorVariable
