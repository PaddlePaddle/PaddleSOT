import unittest

from test_case_base import TestCaseBase

import paddle


def fn(x):
    # now do not support `with xx_guard()`, will fallback to dygraph
    with paddle.static.amp.fp16_guard():
        out = x + 1
        return out


class TestGuard(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor(2)
        self.assert_results(fn, x)


if __name__ == "__main__":
    unittest.main()
