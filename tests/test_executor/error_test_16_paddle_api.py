import unittest

from test_case_base import TestCaseBase

import paddle


def paddle_api_call(x: paddle.Tensor):
    m = x + 2
    m = paddle.nn.functional.relu(m)
    return m


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(paddle_api_call, paddle.to_tensor(2))
        self.assert_results(paddle_api_call, paddle.to_tensor(-5))
        self.assert_results(paddle_api_call, paddle.to_tensor(0))


if __name__ == "__main__":
    unittest.main()
