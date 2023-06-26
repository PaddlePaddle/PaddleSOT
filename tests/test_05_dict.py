# New Supported Instructions:
# BUILD_MAP (new)
# BUILD_CONST_KEY_MAP (new)

import operator
import unittest

from test_case_base import TestCaseBase

import paddle


def build_map(x: int, y: paddle.Tensor):
    z = {x: y}
    return z[x] + 1


def build_const_key_map(x: int, y: paddle.Tensor):
    z = {1: y, 2: y + 1}
    return z[x] + 1


def dict_get_item_1(x: int, y: paddle.Tensor):
    z = {1: y, 2: y + 1}
    return operator.getitem(z, 1)


def dict_get_item_2(x: int, y: paddle.Tensor):
    z = {1: y, 2: y + 1}
    return operator.getitem(z, 2)


def dict_set_item_1(x: int, y: paddle.Tensor):
    z = {1: y, 2: y + 1}
    z[1] = y * 2
    return z[1]


def dict_set_item_2(x: int, y: paddle.Tensor):
    z = {'x': x, 'y': y}
    operator.setitem(z, 'x', 2)
    return z


def dict_set_item_3(x: int, y: paddle.Tensor):
    z = {'x': x, 'y': y}
    operator.setitem(z, 'y', paddle.to_tensor(3))
    return z


def dict_del_item_1(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    del z[1]
    return z


def dict_del_item_2(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    del z[2]
    return z


def dict_del_item_3(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    operator.delitem(z, 1)
    return z


def dict_del_item_4(x: int, y: paddle.Tensor):
    z = {1: x, 2: y + 1}
    operator.delitem(z, 2)
    return z


class TestExecutor(TestCaseBase):
    def test_build_map(self):
        self.assert_results(build_map, 1, paddle.to_tensor(2))

    def test_build_const_key_map(self):
        self.assert_results(build_const_key_map, 1, paddle.to_tensor(2))

    def test_dict_get_item(self):
        self.assert_results(dict_get_item_1, 1, paddle.to_tensor(2))
        self.assert_results(dict_get_item_2, 1, paddle.to_tensor(2))

    def test_dict_set_item(self):
        self.assert_results(dict_set_item_1, 1, paddle.to_tensor(2))
        self.assert_results(dict_set_item_2, 1, paddle.to_tensor(2))
        # TODO(SigureMo) SideEffects have not been implemented yet, we need to skip them
        # self.assert_results(dict_set_item_3, 1, paddle.to_tensor(2))

    def test_dict_del_item(self):
        self.assert_results(dict_del_item_1, 1, paddle.to_tensor(2))
        self.assert_results(dict_del_item_2, 1, paddle.to_tensor(2))
        self.assert_results(dict_del_item_3, 1, paddle.to_tensor(2))
        self.assert_results(dict_del_item_4, 1, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
