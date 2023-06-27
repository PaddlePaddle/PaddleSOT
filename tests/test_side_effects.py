from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def dict_setitem(x):
    x[0] = 1
    return x[0]


def dict_delitem(x):
    del x[0]
    return x


def dict_delitem_getitem(a):
    b = a[0]
    del a[0]
    b[0] = 1
    return a, b


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


class TestDictSideEffect(TestCaseBase):
    def test_dict_setitem(self):
        self.assert_results_with_side_effects(
            dict_setitem, {0: paddle.to_tensor(0)}
        )
        self.assert_results_with_side_effects(
            dict_setitem, {0: paddle.to_tensor(1)}
        )

    def test_dict_delitem(self):
        self.assert_results_with_side_effects(
            dict_delitem, {0: paddle.to_tensor(0), 1: paddle.to_tensor(1)}
        )
        self.assert_results_with_side_effects(
            dict_delitem, {0: paddle.to_tensor(1), 2: paddle.to_tensor(2)}
        )

    def test_dict_delitem_getitem(self):
        self.assert_results_with_side_effects(
            dict_delitem_getitem, {0: {0: 1, 1: 2}}
        )


class TestListSideEffect(TestCaseBase):
    # TODO(SigureMo): Support list side effects.
    def error_test_slice_in_for_loop(self):
        x = 2
        self.assert_results_with_side_effects(slice_in_for_loop, x)


if __name__ == "__main__":
    unittest.main()
