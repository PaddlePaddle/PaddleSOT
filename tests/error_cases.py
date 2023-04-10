import paddle
from symbolic_trace import symbolic_trace
import unittest
from test_case_base import TestCaseBase

def case_list_comp(x):
    def add_one(x):
        return x + 1
    ret = [ add_one(x) for x in x ]
    return ret

def case_map_structure(x):
    def add_one(x):
        return x + 1
    ret = paddle.utils.map_structure(add_one, x)
    return ret

class Test(TestCaseBase):
    def test(self):
        self.assert_results(case_list_comp, paddle.to_tensor([1, 2, 4, 3]))
        self.assert_results(case_map_structure, paddle.to_tensor([1, 2, 4, 3]))

if __name__ == "__main__":
    unittest.main()

