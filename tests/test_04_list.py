# New Supported Instructions:
# BUILD_LIST (new)
# BINARY_SUBSCR
# DELETE_SUBSCR


import operator
import unittest

from test_case_base import TestCaseBase

import paddle


def foo(x: int, y: paddle.Tensor):
    x = [x, y]
    return x[1] + 1


def list_getitem(x: int, y: paddle.Tensor):
    z = [x, y]
    return operator.getitem(z, 1) + 1


def list_setitem_1(x: int, y: paddle.Tensor):
    z = [x, y]
    z[0] = 3
    return z


def list_setitem_2(x: int, y: paddle.Tensor):
    z = [x, y]
    operator.setitem(z, 0, 3)
    return z


def list_setitem_3(x: int, y: paddle.Tensor):
    z = [x, y]
    operator.setitem(z, 1, paddle.to_tensor(3))
    return z


def list_delitem_1(x: int, y: paddle.Tensor):
    z = [x, y]
    del z[0]
    return z


def list_delitem_2(x: int, y: paddle.Tensor):
    z = [x, y]
    del z[1]
    return z


def list_delitem_3(x: int, y: paddle.Tensor):
    z = [x, y]
    operator.delitem(z, 0)
    return z


def list_delitem_4(x: int, y: paddle.Tensor):
    z = [x, y]
    operator.delitem(z, 1)
    return z


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))
        self.assert_results(list_getitem, 1, paddle.to_tensor(2))
        self.assert_results(list_setitem_1, 1, paddle.to_tensor(2))
        self.assert_results(list_setitem_2, 1, paddle.to_tensor(2))
        # TODO(SigureMo) SideEffects have not been implemented yet, we need to skip them
        # self.assert_results(list_setitem_3, 1, paddle.to_tensor(2))
        self.assert_results(list_delitem_1, 1, paddle.to_tensor(2))
        self.assert_results(list_delitem_2, 1, paddle.to_tensor(2))
        self.assert_results(list_delitem_3, 1, paddle.to_tensor(2))
        self.assert_results(list_delitem_4, 1, paddle.to_tensor(2))


if __name__ == "__main__":
    unittest.main()
