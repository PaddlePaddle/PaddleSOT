# New Supported Instructions:
# BUILD_MAP (new)
# BUILD_CONST_KEY_MAP (new)

import unittest

from test_case_base import TestCaseBase

import paddle


def build_map(x: int, y: paddle.Tensor):
    z = {x: y}
    return z[x] + 1


def build_const_key_map(x: int, y: paddle.Tensor):
    z = {1: y, 2: y + 1}
    return z[x] + 1


class TestExecutor(TestCaseBase):
    def test_build_map(self):
        self.assert_results(build_map, 1, paddle.to_tensor(2))

    def test_build_const_key_map(self):
        self.assert_results(build_const_key_map, 1, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
