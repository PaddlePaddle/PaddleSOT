import unittest

from test_case_base import TestCaseBase

import paddle


def dispatch_len(x: paddle.Tensor):
    return len(x.shape)


class TestBuiltinDispatch(TestCaseBase):
    def test_dispatch_len(self):
        self.assert_results(dispatch_len, paddle.to_tensor([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
