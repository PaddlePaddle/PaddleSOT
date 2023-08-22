from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle

global_x = 1
global_y = paddle.to_tensor(2)
global_z = None


def gloabl_func_int():
    global global_x
    global_x = global_x + 1
    return global_x


def gloabl_func_int_add():
    global global_x
    global_x = global_x + global_x
    return global_x + global_x


def gloabl_func_tensor_int_add(tensor_y: paddle.Tensor):
    global global_x
    global_x += 1
    return global_x + tensor_y


def gloabl_func_tensor():
    global global_y
    global_y = global_y + global_y
    return global_y


def gloabl_func_tensor_add():
    global global_y
    global_y = global_y + global_y
    return global_y + global_y


def global_func():
    global global_x
    global global_y
    global global_z

    global_z = global_x + global_y
    return global_z


def global_del_int():
    global global_x
    global global_y

    del global_x
    del global_y
    # return global_x


class TestGlobal(TestCaseBase):
    def test_global_func_int(self):
        global global_x
        self.assert_results_with_global_check(gloabl_func_int, ["global_x"])
        global_x += 1
        self.assert_results_with_global_check(gloabl_func_int, ["global_x"])
        self.assert_results_with_global_check(gloabl_func_int_add, ["global_x"])

    def test_gloabl_func_tensor_int_add(self):
        self.assert_results_with_global_check(
            gloabl_func_tensor_int_add, ["global_x"], paddle.to_tensor(1)
        )

    def test_global_func_tensor(self):
        self.assert_results_with_global_check(gloabl_func_tensor, ["global_y"])
        self.assert_results_with_global_check(
            gloabl_func_tensor_add, ["global_y"]
        )

    def test_global_func(self):
        self.assert_results_with_global_check(global_func, ["global_z"])
        # There are currently no tests for errors
        # self.assert_results_error(global_del_int, KeyError)


if __name__ == "__main__":
    unittest.main()
