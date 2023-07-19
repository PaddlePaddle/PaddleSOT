# TODO: remove
import unittest

from test_case_base import TestCaseBase, strict_mode_guard

import paddle


def foo(x: int, y: paddle.Tensor):
    return x + y / 0


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))


def numpy_add(x, y):
    out = paddle.to_tensor(x.numpy() + y.numpy())
    return out


class TestNumpyAdd(TestCaseBase):
    @strict_mode_guard(0)
    def test_msg1(self):
        x = paddle.to_tensor([2])
        y = paddle.to_tensor([3])
        self.assert_results(numpy_add, x, y)

    def test_msg2(self):
        self.assert_nest_match(1, 2)

    def test_msg3(self):
        self.assert_nest_match(1.0, 1)


if __name__ == "__main__":
    unittest.main()
