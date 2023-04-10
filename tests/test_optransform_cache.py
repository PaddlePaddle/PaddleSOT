import paddle
from symbolic_trace import symbolic_trace
import unittest
from test_case_base import TestCaseBase
from symbolic_trace.opcode_translator.instruction_translator import InstructionTranslatorCache

def case1(x):
    for i in range(int(x)):
        print("yes")
    return x

class TestFor(TestCaseBase):
    def test(self):
        symbolic_trace(case1)(paddle.to_tensor([4]))
        symbolic_trace(case1)(paddle.to_tensor([4]))
        symbolic_trace(case1)(paddle.to_tensor([4]))
        assert InstructionTranslatorCache().hit_num == 2, "cache hit num should be 2"

if __name__ == "__main__":
    unittest.main()
