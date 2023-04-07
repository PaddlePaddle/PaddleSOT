import paddle
from symbolic_trace import symbolic_trace


import paddle
from symbolic_trace import symbolic_trace
import unittest
from test_case_base import TestCaseBase

def case1(x):
    y = 2 + x
    ret = paddle.nn.functional.relu(y)
    print("yes")
    print("no")
    #for i in range(10):
    ret = ret + 2 + x
    return ret

def case2(x):
    y = x + 2 
    ret = paddle.nn.functional.relu(y)
    for i in range(10):
        ret = ret + 2 + x
    return ret

class TestIf(TestCaseBase):
    def test_if_1(self):
        x = paddle.to_tensor([1.0])
        self.assert_results(case1, paddle.to_tensor([1.0]))
        self.assert_results(case2, paddle.to_tensor([1.0]))

if __name__ == "__main__":
    unittest.main()
