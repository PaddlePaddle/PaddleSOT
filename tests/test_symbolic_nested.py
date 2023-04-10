import paddle
from symbolic_trace import symbolic_trace
import unittest
from test_case_base import TestCaseBase

def add_one(x):
    ret = x + 1
    tmp = x + 2
    print(ret)
    return tmp

def case1(x):
    y = x + 1
    x = add_one(x)
    return x + y

def case_map(x):
    def add_one(x):
        return x + 1
    ret = list(map(add_one, x))
    return ret

class Test(TestCaseBase):
    def test(self):
        self.assert_results(case1, paddle.to_tensor([1, 1, 1, 1]))
        self.assert_results(case_map, paddle.to_tensor([1, 1, 1, 1]))

if __name__ == "__main__":
    unittest.main()
