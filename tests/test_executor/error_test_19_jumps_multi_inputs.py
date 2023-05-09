import unittest

from test_case_base import TestCaseBase

import paddle

# TODO(@wangzhen45): if-else fallback support outputs analysis.


def multi_output(x: paddle.Tensor):
    m = x + 1
    if x > 0:
        return m.mean()
    else:
        return 2 * m


class TestExecutor(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor(2)
        self.assert_results(multi_output, x)
        x = paddle.to_tensor(-1)
        self.assert_results(multi_output, x)


if __name__ == "__main__":
    unittest.main()
