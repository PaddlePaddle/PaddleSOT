import unittest

import paddle
from sot import symbolic_translate
from sot.proxy_tensor import cache_and_return, frame_enter, frame_leave


class A:
    def __init__(self, x):
        self.x = x


def sum_2(l):
    if frame_enter("func1", (l)):
        print("hit cache")
        return cache_and_return("func1", (l))
    ret = l[0] + l[1]
    frame_leave(ret)
    return ret


class TestCaseName(unittest.TestCase):
    def test_return_callable(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        ret = symbolic_translate(sum_2)([x, y])
        print(ret)


if __name__ == "__main__":
    unittest.main()
