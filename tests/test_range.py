import unittest

from test_case_base import TestCaseBase

import paddle


def test_range_1(stop: int):
    return range(stop)


def test_range_2(start: int, stop: int):
    return range(start, stop)


def test_range_3(start: int, stop: int, step: int):
    return range(start, stop, step)


def test_range_4(stop: int, index: int):
    return range(stop)[index]


def test_range_5(stop: int):
    return list(range(stop))


def test_range_6(stop: int, index: int):
    return list(range(stop))[index]


def test_range_7(index: int, tensor: paddle.Tensor):
    return list(range(len(tensor.shape)))[index]


def test_range_8(stop: int):
    sum = 0
    for i in range(stop):
        sum += i
    return sum


def test_range_9(stop: int, tensor: paddle.Tensor):
    for i in range(stop):
        tensor += i
    return tensor


class TestExecutor(TestCaseBase):
    def test_cases(self):
        start = 10
        stop = 50
        step = 2
        index = 1
        tensor = paddle.randn((10, 10))
        layer_list = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for _ in range(3)]
        )

        # self.assert_results(test_range_1, stop)
        # self.assert_results(test_range_2, start, stop)
        # self.assert_results(test_range_3, start, stop, step)
        # self.assert_results(test_range_4, stop, index)
        # self.assert_results(test_range_5, stop)
        self.assert_results(test_range_6, stop, index)
        # self.assert_results(test_range_7, index, tensor)
        # self.assert_results(test_range_8, stop)

        # self.assert_results(test_range_9, stop, paddle.randn((10,)))


if __name__ == "__main__":
    unittest.main()
