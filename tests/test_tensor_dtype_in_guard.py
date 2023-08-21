import unittest

from test_case_base import TestCaseBase

import paddle
import sot


def foo(x, y):
    if x.dtype == paddle.float32:
        out = x + y
    else:
        out = x - y
    return out


@sot.skip_function
def dtype_in_guard(x, y):
    with paddle.amp.auto_cast(level='O2'):
        for i in range(10):
            z = foo(x, y)
            x = z
        return x


def dtype_as_input(x, y):
    if x == paddle.float32:
        return y + 1
    else:
        return y - 1


class TestDtypeInGuard(TestCaseBase):
    def test_dtype_in_guard(self):
        x = paddle.to_tensor([2], dtype="float32")
        y = paddle.to_tensor([3], dtype="float32")
        self.assert_results(dtype_in_guard, x, y)

    def test_input_dtype_in_guard(self):
        x = paddle.float32
        y = paddle.to_tensor([3], dtype="float32")
        self.assert_results(dtype_as_input, x, y)


if __name__ == "__main__":
    unittest.main()
