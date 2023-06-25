import unittest

from test_case_base import TestCaseBase

import paddle


def slice_in_for_loop(x, iter_num=3):
    x = paddle.to_tensor(x)
    a = []

    iter_num = paddle.full(shape=[1], fill_value=iter_num, dtype="int32")

    for i in range(iter_num):
        a.append(x)

    for i in range(iter_num):
        a[i] = x
    out = a[2]
    return out


class TestListSideEffect(TestCaseBase):
    def test_slice_in_for_loop(self):
        x = 2
        self.assert_results(slice_in_for_loop, x)


if __name__ == "__main__":
    unittest.main()
