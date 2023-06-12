import inspect
import unittest

import paddle
from sot.opcode_translator.instruction_utils import (
    analysis_inputs,
    calc_offset_from_bytecode_offset,
    get_instructions,
)


def assert_inputs_equals(expected_inputs):
    current_frame = inspect.currentframe()
    assert current_frame is not None
    test_frame = current_frame.f_back
    assert test_frame is not None

    instructions = get_instructions(test_frame.f_code)
    current_instr_idx = calc_offset_from_bytecode_offset(test_frame.f_lasti)
    actual_inputs = analysis_inputs(instructions, current_instr_idx)
    assert (
        actual_inputs == expected_inputs
    ), f"actual_inputs: {actual_inputs}, expected_inputs: {expected_inputs}"


def case1(x):
    m = x + 1
    n = x + 2
    assert_inputs_equals({"x", "n"})
    y = x + 2
    assert_inputs_equals({"n"})
    return n


def case2(x):
    x = x + 1
    assert_inputs_equals({"x"})
    y = x + 3
    z = x + y
    assert_inputs_equals({"x"})
    x += 1
    m = x + 1
    n = x + m
    assert_inputs_equals(set())
    return 1


def case3(x):
    y = x + 1

    assert_inputs_equals({"x"})
    if x:
        z = 1
    else:
        z = 2
    return z


def case4(x):
    y = x + 1

    assert_inputs_equals({"x", "y"})
    if x:
        z = y
    else:
        z = x
    return z


def case5(x):
    y = x + 1
    z = x + 2

    assert_inputs_equals({"z"})
    if z:
        a = 1
    else:
        b = 2
    return z


def case6(x):
    y = x + 1
    z = x + 2

    assert_inputs_equals({"a", "z"})
    if z:
        a = 1
    else:
        a += 1
    return z


def case7(x):
    y = x + 1
    z = x + 2

    assert_inputs_equals({"a", "z"})
    if not z:
        a += 1  # noqa: F821
    else:
        a = 1
    return z


class TestAnalysisInputs(unittest.TestCase):
    def test_case1(self):
        case1(paddle.to_tensor([1]))

    def test_case2(self):
        case2(paddle.to_tensor([2]))

    def test_case3(self):
        case3(paddle.to_tensor([3]))

    def test_case4(self):
        case4(paddle.to_tensor([4]))

    def test_case5(self):
        case5(paddle.to_tensor([5]))

    def test_case6(self):
        case6(paddle.to_tensor([6]))

    # def test_case7(self):
    #     case7(paddle.to_tensor([7]))


if __name__ == "__main__":
    unittest.main()
