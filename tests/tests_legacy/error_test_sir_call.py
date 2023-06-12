import unittest

import paddle
from sot import symbolic_translate
from sot.trace_cache_entrance import cache_and_return, frame_enter, frame_leave


def sum(x, y):
    if frame_enter("sum", (x, y)):
        return cache_and_return("sum", (x, y))
    ret = x + y
    frame_leave("sum", (ret))
    return ret


def main(x, y):
    ret = sum(x, x)
    ret2 = sum(x, y)
    return ret2


class TestCaseName(unittest.TestCase):
    def test_return_callable(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        ret = symbolic_translate(main)(x, y)
        assert ret.item() == 3.0, "Should be 4.0"


if __name__ == "__main__":
    unittest.main()
