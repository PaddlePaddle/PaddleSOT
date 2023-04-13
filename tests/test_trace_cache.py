import unittest
import paddle
from symbolic_trace import symbolic_trace
from symbolic_trace.proxy_tensor import frame_enter, frame_leave, cache_and_return



def case1(x):
    x = func1(x)                # cache SIR
    x, z = func2(x, "ok")       # cache SIR
    x = x + 5
    x = func1(x)                # hit cache
    x, z = func2(x, "ok")       # hit cache
    x, z = func2(x, "no")       # cache SIR
    return x

def func1(x):
    if frame_enter("func1", (x)):
        return cache_and_return("func1", (x))
    ret = x + 2
    frame_leave((ret))
    return ret

def func2(x, string):
    if frame_enter("func2", (x, string)):
        return cache_and_return("func2", (x, string))
    x = x * 2
    frame_leave((x, string))
    return x, string


class TestCaseName(unittest.TestCase):
    def test_return_callable(self):
        x = paddle.to_tensor([1.0])
        ret = symbolic_trace(case1)(x)
        assert TraceCache().hit_num == 2

if __name__ == "__main__":
    unittest.main()
