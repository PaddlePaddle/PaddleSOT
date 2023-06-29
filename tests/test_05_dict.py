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


def dict_set_item_int(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    z[1] = x * 2
    return z[1]


def dict_set_item_tensor(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    z[2] = paddle.to_tensor(4)
    return z[1]


def dict_del_item_int(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    del z[1]
    return z


def dict_del_item_tensor(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    del z[2]
    return z


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


class TestExecutor(TestCaseBase):
    def test_build_map(self):
        self.assert_results(build_map, 1, paddle.to_tensor(2))

    def test_build_const_key_map(self):
        self.assert_results(build_const_key_map, 1, paddle.to_tensor(2))

    def test_dict_set_item(self):
        self.assert_results(dict_set_item_int, 1, paddle.to_tensor(2))
        self.assert_results(dict_set_item_tensor, 1, paddle.to_tensor(2))

    def test_dict_del_item(self):
        self.assert_results(dict_del_item_int, 1, paddle.to_tensor(2))
        self.assert_results(dict_del_item_tensor, 1, paddle.to_tensor(2))

    def test_construct(self):
        self.assert_results(dict_construct_from_dict)
        self.assert_results(dict_construct_from_list)
        self.assert_results(dict_construct_from_tuple)


if __name__ == "__main__":
    unittest.main()
