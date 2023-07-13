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


def gloabl_func_tensor():
    global global_y
    global_y = global_y + global_y
    return global_y


def global_func():
    global global_x
    global global_y
    global global_z

    global_z = global_x + global_y
    return global_z


class TestExecutor(TestCaseBase):
    def test_global_func_int(self):
        self.assert_results(gloabl_func_int)

    # def test_global_func_tensor(self):
    #     self.assert_results(gloabl_func_tensor)

    # def test_global_func(self):
    #     self.assert_results(global_func)


if __name__ == "__main__":
    unittest.main()
