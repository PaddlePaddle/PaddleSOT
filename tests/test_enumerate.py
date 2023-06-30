import unittest

from test_case_base import TestCaseBase

import paddle


def test_enumerate_1(x: int, y: int):
    for id, val in enumerate(range(x)):
        if id % 2 == 0:
            y += val
    return y


def test_enumerate_2(x: list):
    return list(enumerate(x))


def test_enumerate_3(x: paddle.Tensor):
    sum = 0
    for idx, val in enumerate(x):
        sum += val
    return sum


def test_enumerate_4(layer_list, x):
    sum = 0
    for idx, layer in enumerate(layer_list):
        sum += layer(x)
    return sum


class TestExecutor(TestCaseBase):
    def test_cases(self):
        x = 8
        y = 5
        ty = paddle.randn((10, 10))
        layer_list = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for _ in range(3)]
        )

        # self.assert_results(test_enumerate_1, x, y)
        # self.assert_results(test_enumerate_2, [2, 4, 6, 8, 10])
        self.assert_results(test_enumerate_3, paddle.randn((10,)))
        # self.assert_results(test_enumerate_4, layer_list, paddle.randn((10,)))


if __name__ == "__main__":
    unittest.main()
