# New Supported Instructions:
# BUILD_MAP (new)
# BUILD_CONST_KEY_MAP (new)

import unittest

from test_case_base import TestCaseBase

import paddle


def func_1(format_str, tensor):
    str = format_str.format(xx=12)
    a = "{xx} = 12".format
    ttt = f"{10} = 12"
    a(xx=12)
    tensor = tensor + 1
    return str, tensor


def func_2(format_str, tensor):
    str = format_str % 10
    tensor = tensor + 1
    return str, tensor


class TestConstantGraph(TestCaseBase):
    def test_case_1(self):
        x = "{xx} is xx"
        tensor = paddle.to_tensor(1)
        self.assert_results(func_1, x, tensor)

    def test_case_2(self):
        x = "%s is xx"
        tensor = paddle.to_tensor(1)
        self.assert_results(func_2, x, tensor)


if __name__ == "__main__":
    unittest.main()
