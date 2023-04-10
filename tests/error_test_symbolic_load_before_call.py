import paddle
import unittest
from test_case_base import TestCaseBase

# TODO(SigureMo): Fix this test case

def nest_func_3(x, y):
    a = x + y
    b = x * a
    z = b + nest_func_2(a)
    return z

def nest_func_2(z):
    print(z)
    return z

class Test(TestCaseBase):
    def test(self):
        self.assert_results(
            nest_func_3,
            paddle.to_tensor([1, 1, 1, 1]),
            paddle.to_tensor([2, 3, 4, 5]),
        )

if __name__ == "__main__":
    unittest.main()
