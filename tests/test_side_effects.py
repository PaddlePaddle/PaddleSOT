from __future__ import annotations

import unittest

from test_case_base import TestCaseBase, strict_mode_guard

import paddle
from sot import symbolic_translate
from sot.utils import InnerError


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


def dict_nested_1(x):
    x[0][0] = 42
    x[1][0] = x[0][0] + x[0][1]
    x[2] = {1: 2}
    return x


def dict_nested_2(x):
    a = x[0]
    b = x[1]
    del a[0]
    a[1] = b[0]
    a[2] = b[1]
    x[1][0] = 42
    del a[1]
    return a, b


def list_append_int(tensor_x, list_a):
    tensor_x = tensor_x + 1
    list_a.append(12)
    return tensor_x, list_a


def list_append_tensor(tensor_x, list_a):
    tensor_x = tensor_x + 1
    list_a.append(tensor_x)
    return tensor_x, list_a


def list_delitem(list_a):
    del list_a[0]
    return list_a[0]


def list_extend(list_a):
    list_a.extend([1, 2, 3])
    return list_a[0]


def list_nested(list_a):
    inner_list = []
    inner_list.append(list_a)
    inner_list[-1].append(12)
    return 12


def list_insert(list_a):
    list_a.insert(0, 1)
    return list_a[0]


def list_remove(list_a):
    list_a.remove(1)
    return list_a[0]


def list_pop(list_a):
    list_a.pop(0)
    list_a.pop()
    list_a.pop(1)
    return list_a[0]


def list_clear(list_a):
    list_a.clear()
    return list_a


def list_sort(list_a):
    list_a.sort()
    return list_a


def list_reverse(list_a):
    list_a.reverse()
    return list_a


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


# TODO: Global SideEffect
a = 12


def normal_size_effect_6(tensor_x):
    """global"""
    global a
    a = 1
    return tensor_x + a


# TODO: Object SideEffect
class CustomObject:
    def __init__(self):
        self.x = 0


def object_attr(cus_obj, t):
    """object side effect."""
    t = t + 1
    cus_obj.x = t
    return t, cus_obj


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

    def test_dict_nested_1(self):
        self.assert_results_with_side_effects(
            dict_nested_1, {0: {0: 1, 1: 2}, 1: {0: 1, 1: 2}}
        )
        self.assert_results_with_side_effects(
            dict_nested_1, {0: {0: 123, 1: 2}, 1: {0: 1, 1: 2}}
        )

    def test_dict_nested_2(self):
        self.assert_results_with_side_effects(
            dict_nested_2, {0: {0: 1, 1: 2}, 1: {0: 1, 1: 2}}
        )
        self.assert_results_with_side_effects(
            dict_nested_2, {0: {0: 123, 1: 2}, 1: {0: 1, 1: 2}}
        )


class TestListSideEffect(TestCaseBase):
    def test_list_append(self):
        self.assert_results_with_side_effects(
            list_append_int, paddle.to_tensor(1), [1, 2, 3]
        )
        self.assert_results_with_side_effects(
            list_append_tensor, paddle.to_tensor(2), [1, 2, 3]
        )

    def test_list_delitem(self):
        self.assert_results_with_side_effects(list_delitem, [1, 2, 3])

    def test_list_extend(self):
        self.assert_results_with_side_effects(
            list_extend, [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

    def test_list_insert(self):
        self.assert_results_with_side_effects(list_insert, [1, 2, 3])
        self.assert_results_with_side_effects(
            list_insert, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    def test_list_remove(self):
        self.assert_results_with_side_effects(list_remove, [1, 1, 1])
        self.assert_results_with_side_effects(list_remove, [0, 1, 2])
        with self.assertRaises(InnerError):
            symbolic_translate(list_remove)([0, 2, 4])

    def test_list_pop(self):
        self.assert_results_with_side_effects(list_pop, [1, 2, 3, 4, 5])
        self.assert_results_with_side_effects(
            list_pop, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    def test_list_clear(self):
        self.assert_results_with_side_effects(list_clear, [1, 2, 3, 4, 5])
        self.assert_results_with_side_effects(
            list_clear, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    def test_list_sort(self):
        self.assert_results_with_side_effects(list_sort, [2, 1, 7, 3, 4, 6])
        self.assert_results_with_side_effects(
            list_sort, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    def test_list_reverse(self):
        self.assert_results_with_side_effects(list_reverse, [1, 2, 3, 4, 5])
        self.assert_results_with_side_effects(
            list_reverse, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    def test_slice_in_for_loop(self):
        x = 2
        with strict_mode_guard(0):
            self.assert_results_with_side_effects(slice_in_for_loop, x)

    def test_list_nested(self):
        self.assert_results_with_side_effects(list_nested, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
