import paddle
from symbolic_trace import symbolic_trace
import unittest
from test_case_base import TestCaseBase

def case1(cond, x):
    cond = 2 + cond
    if cond:
        print('yes')
        x = x + 4
    else:
        print("no")
        x = x - 4
    ret = paddle.nn.functional.relu(x)
    return ret

class TestIf(TestCaseBase):
    def test_if_1(self):
        self.assert_results(case1, paddle.to_tensor([4.0]), paddle.to_tensor([4.0]))
        self.assert_results(case1, paddle.to_tensor([-2.0]), paddle.to_tensor([4.0]))

if __name__ == "__main__":
    unittest.main()
