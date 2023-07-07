# New Supported Instructions:
# BUILD_LIST (new)
# BINARY_SUBSCR
# DELETE_SUBSCR


import unittest

from test_case_base import TestCaseBase

import paddle


def list_getitem_int(x: int, y: paddle.Tensor):
    x = [x, y]
    return x[0] + 1


def list_getitem_tensor(x: int, y: paddle.Tensor):
    x = [x, y]
    return x[1] + 1


def list_setitem_int(x: int, y: paddle.Tensor):
    z = [x, y]
    z[0] = 3
    return z


def list_setitem_tensor(x: int, y: paddle.Tensor):
    z = [x, y]
    z[1] = paddle.to_tensor(3)
    return z


def list_delitem_int(x: int, y: paddle.Tensor):
    z = [x, y]
    del z[0]
    return z


def list_delitem_tensor(x: int, y: paddle.Tensor):
    z = [x, y]
    del z[1]
    return z


def list_append_int(x: int, y: paddle.Tensor):
    z = [x, y]
    z.append(3)
    return z


def list_append_tensor(x: int, y: paddle.Tensor):
    z = [x, y]
    z.append(y)
    return z


def list_clear(x: int, y: paddle.Tensor):
    z = [x, y]
    z.clear()
    return z


def list_copy(x: int, y: paddle.Tensor):
    z = [x, y]
    a = z.copy()
    z[0] = 3
    z[1] = y + 1
    return (a, z)


def list_count(x: int, y: paddle.Tensor):
    z = [x, x, y, y, y]
    return (z.count(x), z.count(y))


def list_extend(x: int, y: paddle.Tensor):
    z = [x, y]
    a = [y, x]
    b = (x, y)
    z.extend(a)
    z.extend(b)
    return z


def list_index(x: int, y: paddle.Tensor):
    z = [y, x, y, y]
    return (z.index(x), z.index(y))


def list_insert(x: int, y: paddle.Tensor):
    z = [x, y]
    z.insert(0, x)
    z.insert(3, y)
    return z


def list_pop(x: int, y: paddle.Tensor):
    z = [x, y]
    a = z.pop()
    b = z.pop()
    return (z, a, b)


def list_remove(x: int, y: paddle.Tensor):
    z = [x, x, y, y]
    z.remove(x)
    z.remove(y)
    return z


def list_reverse(x: int, y: paddle.Tensor):
    z = [x, x, y, y]
    z.reverse()
    return z


def list_default_sort(x: int, y: paddle.Tensor):
    z = [y + 2, y, y + 1]
    z.sort()
    return z


def list_key_sort(x: int, y: paddle.Tensor):
    z = [y + 2, y, y + 1]
    z.sort(key=len)
    return z


def list_reverse_sort(x: int, y: paddle.Tensor):
    z = [y + 2, y, y + 1]
    z.sort(reverse=True)
    return z


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(list_getitem_int, 1, paddle.to_tensor(2))
        self.assert_results(list_getitem_tensor, 1, paddle.to_tensor(2))
        self.assert_results(list_setitem_int, 1, paddle.to_tensor(2))
        self.assert_results(list_setitem_tensor, 1, paddle.to_tensor(2))
        self.assert_results(list_count, 1, paddle.to_tensor(2))
        self.assert_results(list_index, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(
            list_delitem_int, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_delitem_tensor, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_append_int, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_append_tensor, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_clear, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(list_copy, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(
            list_extend, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_insert, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(list_pop, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(
            list_remove, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_reverse, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_reverse, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_default_sort, 1, paddle.to_tensor(2)
        )
        # self.assert_results_with_side_effects(
        #     list_default_sort, 1, paddle.to_tensor(2)
        # )


if __name__ == "__main__":
    unittest.main()
