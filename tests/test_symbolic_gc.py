import paddle
import unittest
from test_case_base import TestCaseBase, check_live_vars

def case1(x):
    m = x + 1
    n = x + 2
    print(m)
    check_live_vars(has_value_vars=["x", "n"], no_value_vars=["m"])
    y = x + 2
    print(y)
    check_live_vars(has_value_vars=["n"], no_value_vars=["m", "x", "y"])
    return n

def case2(x):
    x = x + 1
    print(x)
    check_live_vars(has_value_vars=["x"], no_value_vars=[])
    y = x + 3
    z = x + y
    print(y)
    check_live_vars(has_value_vars=["x"], no_value_vars=["y", "z"])
    x += 1
    m = x + 1
    n = x + m
    print(m)
    check_live_vars(has_value_vars=[], no_value_vars=["x", "y", "z", "m", "n"])
    return 1

def case3_called(x):
    y = x + 1
    print(y)
    # x is used in outer function
    check_live_vars(has_value_vars=["x"], no_value_vars=[])
    return 1

def case3(x):
    y = case3_called(x)
    z = x + 1
    return z
    

class TestFor(TestCaseBase):
    def test(self):
        self.assert_results(case1, paddle.to_tensor([4]))
        self.assert_results(case2, paddle.to_tensor([4.0, 1.0, 2.0, 3.0]))
        self.assert_results(case3, paddle.to_tensor([1.0, 2.0]))

if __name__ == "__main__":
    unittest.main()
