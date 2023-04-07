import paddle
from symbolic_trace import symbolic_trace
import unittest
from test_case_base import TestCaseBase

def case1(x):
    for i in range(int(x)):
        print("yes")
    return x

class TestFor(TestCaseBase):
    def test(self):
        x = paddle.to_tensor([4.0])
        self.assert_results(case1, x)

if __name__ == "__main__":
    unittest.main()
