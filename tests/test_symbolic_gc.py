import paddle
import unittest
from symbolic_trace import symbolic_trace
from test_case_base import TestCaseBase, check_live_vars

def case1(x):
    m = x + 1
    n = x + 2
    print(m)
    check_live_vars(live_vars=["x", "n"], dead_vars=["m"])
    y = x + 2
    print(y)
    check_live_vars(live_vars=["n"], dead_vars=["m", "x", "y"])
    return n

def case2(x):
    x = x + 1
    print(x)
    check_live_vars(live_vars=["x"], dead_vars=[])
    y = x + 3
    z = x + y
    print(y)
    check_live_vars(live_vars=["x"], dead_vars=["y", "z"])
    x += 1
    m = x + 1
    n = x + m
    print(m)
    check_live_vars(live_vars=[], dead_vars=["x", "y", "z", "m", "n"])
    return 1

def case3_called(u):
    v = u + 1
    print(v)
    # u is used in outer function (x)
    check_live_vars(live_vars=["u"], dead_vars=["v"])
    return 1

def case3(x):
    y = case3_called(x)
    z = x + 1
    return z

def case4_called(u):
    u = u + 1
    print(u)
    # This u is a new symbol in SIR, it is not the same as the x in outer function
    check_live_vars(live_vars=[], dead_vars=["u"])
    return 1

def case4(x):
    y = case4_called(x)
    z = x + 1
    return z

def case5(x):
    y = x + 1
    x = y
    print(x)
    # x is same as y
    check_live_vars(live_vars=["x", "y"], dead_vars=[])
    x = x + 1
    print(x)
    # x is different from y
    check_live_vars(live_vars=["x"], dead_vars=["y"])
    return x

class TestFor(TestCaseBase):
    def test(self):
        symbolic_trace(case1)(paddle.to_tensor([4]))
        symbolic_trace(case2)(paddle.to_tensor([4.0, 1.0, 2.0, 3.0]))
        symbolic_trace(case3)(paddle.to_tensor([1.0, 2.0]))
        symbolic_trace(case4)(paddle.to_tensor([7.0, 1.0, 2.0]))
        symbolic_trace(case5)(paddle.to_tensor([7.0, 1.0]))

if __name__ == "__main__":
    unittest.main()
