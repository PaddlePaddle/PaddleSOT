# New Supported Instructions:
# BUILD_MAP (new)
# BUILD_CONST_KEY_MAP (new)

import unittest

from test_case_base import TestCaseBase

import paddle


def build_map(x: int, y: paddle.Tensor):
    z = {x: y}
    return z[x] + 1


def build_const_key_map(x: int, y: paddle.Tensor):
    z = {1: y, 2: y + 1}
    return z[x] + 1


def dict_get_item(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    return (z.get(1), z.get(2))


def dict_get_item_default(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    return (z.get(3, 2), z.get(4, y))


def dict_set_item_int(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    z[1] = x * 2
    return z[1]


def dict_set_item_tensor(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    z[2] = paddle.to_tensor(4)
    return z[1]


def dict_update_item1(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    z.update({1: x * 2, 2: y, 3: y + 2})
    return z


def dict_update_item2(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    z.update({1: x * 2, 2: y, 3: z[2] + 2})
    return z


def dict_del_item_int(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    del z[1]
    return z


def dict_del_item_tensor(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    del z[2]
    return z


def dict_clean_item(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    z.clear()
    return z


def dict_copy_item(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    z2 = z.copy()
    z[1] = 2
    return z2


def dict_fromkeys_int(x: int, y: paddle.Tensor):
    z = dict.fromkeys([1, 2, 3], x)
    return z


def dict_fromkeys_tensor(x: int, y: paddle.Tensor):
    z = dict.fromkeys([1, 2, 3], y)
    return z


def dict_fromkeys_nodefault(x: int, y: paddle.Tensor):
    z = dict.fromkeys([1, 2, 3])
    return z


def dict_setdefault_int(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    a = z.setdefault(4)
    b = z.setdefault(1, 2)
    c = z.setdefault(3, 4)
    return (z, a, b, c)


def dict_pop(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1, 3: y}
    a = z.pop(1)
    b = z.pop(2, 3)
    c = z.pop(4, 3)
    d = z.pop(5, y)
    return (z, a, b, c, d)


def dict_popitem(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1, 3: y}
    a = z.popitem()
    return (z, a)


def dict_construct_from_dict():
    x = {1: 2, 3: 4}
    d = dict(x)
    return d


def dict_construct_from_list():
    x = [[1, 2], [3, 4]]
    d = dict(x)
    return d


def dict_construct_from_tuple():
    x = ((1, 2), (3, 4))
    d = dict(x)
    return d


def dict_construct_from_comprehension():
    z = {1: 2, 3: 4}
    d = {k: v + 1 for k, v in z.items()}
    return d


def dict_no_arguments():
    d1 = dict()  # noqa: C408
    d1.update({1: 2})
    d2 = dict()  # noqa: C408
    d2.update({3: 4})
    return d1[1] + d2[3]


class TestDict(TestCaseBase):
    def test_build_map(self):
        self.assert_results(build_map, 1, paddle.to_tensor(2))

    def test_build_const_key_map(self):
        self.assert_results(build_const_key_map, 1, paddle.to_tensor(2))
        self.assert_results(dict_get_item, 1, paddle.to_tensor(2))
        self.assert_results(dict_get_item_default, 1, paddle.to_tensor(2))

    def test_dict_set_item(self):
        self.assert_results_with_side_effects(
            dict_set_item_int, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_set_item_tensor, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_copy_item, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_fromkeys_int, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_fromkeys_tensor, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_fromkeys_nodefault, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_update_item1, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_update_item2, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_setdefault_int, 1, paddle.to_tensor(2)
        )

    def test_dict_del_item(self):
        self.assert_results_with_side_effects(
            dict_del_item_int, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_del_item_tensor, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            dict_clean_item, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(dict_pop, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(
            dict_popitem, 1, paddle.to_tensor(2)
        )

    def test_construct(self):
        self.assert_results(dict_construct_from_dict)
        self.assert_results(dict_construct_from_list)
        self.assert_results(dict_construct_from_tuple)
        self.assert_results(dict_construct_from_comprehension)

    def test_dict_noargs(self):
        self.assert_results(dict_no_arguments)


if __name__ == "__main__":
    unittest.main()
