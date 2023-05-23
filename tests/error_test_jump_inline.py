from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def pop_jump_if_false(x: bool, y: paddle.Tensor):
    if x:
        y += 1
    else:
        y -= 1
    return y


def outter_function(x, y):
    m = y + 2
    ret = pop_jump_if_false(x, m)
    ret = ret * 2
    return ret


a = paddle.to_tensor(1)
b = paddle.to_tensor(2)
c = paddle.to_tensor(3)
d = paddle.to_tensor(4)


true_tensor = paddle.to_tensor(True)
false_tensor = paddle.to_tensor(False)


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(outter_function, True, a)
        self.assert_results(outter_function, False, a)

    def test_fallback(self):
        self.assert_results(outter_function, true_tensor, a)
        self.assert_results(outter_function, false_tensor, a)


if __name__ == "__main__":
    unittest.main()
