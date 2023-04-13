import unittest
import paddle
from symbolic_trace import symbolic_trace
from symbolic_trace.trace_cache_entrance import frame_enter, frame_leave, cache_and_return


def sum(x, y):
    if frame_enter("sum", (x, y)):
        return cache_and_return("sum", (x, y))
    ret = x + y
    frame_leave("sum", (ret))
    return ret

def main(x, y):
    ret =  sum(x, x)
    ret2 = sum(x, y)
    return ret2

class TestCaseName(unittest.TestCase):
    def test_return_callable(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        ret = symbolic_trace(main)(x, y)
        assert (ret.item() == 3.0), "Should be 4.0"

if __name__ == "__main__":
    unittest.main()
