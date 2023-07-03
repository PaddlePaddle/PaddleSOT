import operator
import unittest

from test_case_base import TestCaseBase

import paddle


def dispatch_len(x: paddle.Tensor):
    return len(x.shape)


def dispatch_bool(x: paddle.Tensor):
    return operator.truth(x.shape) and bool(x.shape)


class TestBuiltinDispatch(TestCaseBase):
    def test_dispatch_len(self):
        self.assert_results(dispatch_len, paddle.to_tensor([1, 2, 3]))

    def test_dispatch_bool(self):
        self.assert_results(dispatch_bool, paddle.to_tensor([1, 2, 3]))


def run_getattr(x: paddle.Tensor):
    attr = 'dtype'
    out = getattr(x, attr)
    return out


class TestGetattr(TestCaseBase):
    def test_getattr(self):
        x = paddle.to_tensor(4)
        self.assert_results(run_getattr, x)


if __name__ == "__main__":
    unittest.main()
