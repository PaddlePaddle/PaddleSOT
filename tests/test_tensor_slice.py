import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: paddle.Tensor):
    return x[:, 0]


class TestExecutor(TestCaseBase):
    def test_tensor_slice(self):
        x = paddle.randn((10, 10))
        self.assert_results(foo, x)


if __name__ == "__main__":
    unittest.main()
