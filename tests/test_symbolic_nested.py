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

def nest_func_3(x, y):
    a = x + y
    b = x * a
    c = nest_func_2(x, y, a)
    print(a)
    nest_func_2(x, y, a)
    z = b * x + c
    return z

def nest_func_2(x, y, z):
    a = x * 2
    a = nest_func_1(a)
    print(z)
    b = a + y
    b = nest_func_1(b)
    return b + z

def nest_func_1(x):
    x += 1
    print(x)
    return x + 1

def case2(x, y, z):
    a = x + y
    x = nest_func_3(z, a)
    z += a
    a = nest_func_3(z, a)
    return a

def case_map(x):
    def add_one(x):
        return x + 1
    ret = list(map(add_one, x))
    return ret

class Test(TestCaseBase):
    def test(self):
        self.assert_results(case1, paddle.to_tensor([1, 1, 1, 1]))
        self.assert_results(
            case2,
            paddle.to_tensor([1, 1, 1, 1]),
            paddle.to_tensor([2, 3, 4, 5]),
            paddle.to_tensor([6, 7, 8, 9])
        )
        self.assert_results(case_map, paddle.to_tensor([1, 1, 1, 1]))

if __name__ == "__main__":
    unittest.main()
