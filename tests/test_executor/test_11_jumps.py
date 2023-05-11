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


def jump_if_false_or_pop(x: bool, y: paddle.Tensor):
    return x and (y + 1)


def jump_if_true_or_pop(x: bool, y: paddle.Tensor):
    return x or (y + 1)


def pop_jump_if_true(x: bool, y: bool, z: paddle.Tensor):
    return (x or y) and z


def jump_absolute(x: int, y: paddle.Tensor):
    while x > 0:
        y += 1
        x -= 1
    return y


a = paddle.to_tensor(1)
b = paddle.to_tensor(2)
c = paddle.to_tensor(3)
d = paddle.to_tensor(4)


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(pop_jump_if_false, True, a)
        self.assert_results(jump_if_false_or_pop, True, a)
        self.assert_results(jump_if_true_or_pop, False, a)
        self.assert_results(pop_jump_if_true, True, False, a)
        self.assert_results(jump_absolute, 5, a)


if __name__ == "__main__":
    unittest.main()

# TODO: JUMP_FORWARD
