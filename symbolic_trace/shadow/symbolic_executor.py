from .symbolic_frame import SymbolicFrame
from ..opcode_translator.executor.source import LocalSource
from ..opcode_translator.executor.variables import (
    ConstantVariable,
)
from ..utils import no_eval_frame
import types
import dis
import sys

class SymbolicExecutor:
    frame: SymbolicFrame
    # next instruction to be executed.
    next_instruction_index: int
  
    def __init__(self, frame: SymbolicFrame):
        self.frame = frame
        self.next_instruction_index = 0

    @no_eval_frame
    def __call__(self, instruction_index):
        instruction = self.frame.instructions[instruction_index]
        if self.next_instruction_index != instruction_index:
            self._run_post_jump_instruction(self.next_instruction_index, instruction_index)
        self._run_post_instruction(instruction_index)
        self.next_instruction_index = instruction_index + 1

    @no_eval_frame
    def pre_action(self, instruction_index):
        instruction = self.frame.instructions[instruction_index]
        method_name = f"pre_{instruction.opname}"
        assert hasattr(self, method_name)
        getattr(self, method_name)(instruction)

    def pre_RETURN_VALUE(self, instruction):
        raise NotImplementedError("Derived class should override prev_RETURN_VALUE() method")

    def _run_post_jump_instruction(self, jump_instruction_index, target_instruction_index):
        jump_instruction = self.get_instruction(jump_instruction_index)
        assert self._is_jump_instruction(jump_instruction)
        is_jump = self._is_jump(jump_instruction, target_instruction_index)
        TODO

    def _run_post_instruction(self, instruction_index):
        assert instruction_index >= 0
        instruction = self.frame.instructions[instruction_index]
        opname = instruction.opname
        assert hasattr(self, opname), f"{opname} not supported"
        method = getattr(self, opname)
        method(instruction)
  
    def push(self, value):
        self.frame.stack.append(value)
  
    def pop(self):
        return self.frame.stack.pop()
  
    def LOAD_FAST(self, instr): 
        varname = instr.argval
        var = self.frame.f_locals[varname]
        var.try_set_source(LocalSource(instr.arg, varname))
        self.push(var)

    def STORE_FAST(self, instr):
        """
        TODO: side effect may happen
        """
        var = self.pop()
        self.frame.f_locals[instr.argval] = var

    def LOAD_CONST(self, instr):
        var = ConstantVariable(instr.argval)
        self.push(var)
  
    def BINARY_ADD(self, instr):
        b = self.pop()
        a = self.pop()
        self.push(a + b)

    def BINARY_MULTIPLY(self, instr):
        b = self.pop()
        a = self.pop()
        self.push(a * b)

    def RETURN_VALUE(self, instr):
        raise NotImplementedError("dead code never to be executed.")

    def __del__(self):
        # Do nothing.
        pass
