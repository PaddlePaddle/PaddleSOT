import paddle
from symbolic_trace import symbolic_trace
import unittest
from test_case_base import TestCaseBase

def case1(x):
    kk = 'sdfsdf'
    y = 2 + x
    ret = paddle.nn.functional.relu(y)
    print(y)
    print(y.numpy())
    print("yes")
    print("no")
    #for i in range(10):
    ret = ret + 2 + x
    return ret

class TestFallback(TestCaseBase):
    def test_bool(self):
        x = paddle.to_tensor([1.0])
        self.assert_results(case1, x)

if __name__ == "__main__":
    unittest.main()
