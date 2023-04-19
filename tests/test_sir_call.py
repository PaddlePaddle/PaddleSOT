import unittest
import paddle
from symbolic_trace import symbolic_trace
from symbolic_trace.trace_cache_entrance import trace_cache


@trace_cache
def sum(x, y):
    ret = x + y
    return ret

def main(x, y):
    ret = sum(x, y=x)
    ret2 = sum(x, y)
    return ret2

class TestCaseName(unittest.TestCase):
    def test_return_callable(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        ret = symbolic_trace(main)(x, y)
        assert (ret.item() == 3.0), "Should be 3.0"

if __name__ == "__main__":
    unittest.main()
