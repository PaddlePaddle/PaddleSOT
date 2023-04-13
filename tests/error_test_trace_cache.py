import unittest
import paddle
from symbolic_trace import symbolic_trace
from symbolic_trace.proxy_tensor import frame_enter, frame_leave, cache_and_return, TraceCache


class A:
    def __init__(self, x):
        self.x = x

def sum_2(l):
    if frame_enter("func1", (l)):
        print('hit cache')
        return cache_and_return("func1", (l))
    ret = l[0] + l[1]
    frame_leave((ret))
    return ret


class TestCaseName(unittest.TestCase):
    def test_return_callable(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        ret = symbolic_trace(sum_2)([x, y])
        print(ret)

if __name__ == "__main__":
    unittest.main()
