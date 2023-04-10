import paddle
from symbolic_trace import symbolic_trace
import unittest
from test_case_base import TestCaseBase

def case1(x):
    for i in range(int(x)):
        print("yes")
    return x

def case2(x):
    sum = 0
    for i in x:
        sum += i
    return sum

class TestFor(TestCaseBase):
    def test(self):
        self.assert_results(case1, paddle.to_tensor([4]))
        self.assert_results(case2, paddle.to_tensor([4.0, 1.0, 2.0, 3.0]))

if __name__ == "__main__":
    unittest.main()
