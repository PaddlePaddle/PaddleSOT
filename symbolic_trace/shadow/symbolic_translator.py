import types
from typing import List
import dis
from . import symbolic_frame_stack as symbolic_frame_stack
from .symbolic_frame_mgr import SymbolicFrameMgr
from ..opcode_translator.executor.pycode_generator import PyCodeGen
from .symbolic_executor import SymbolicExecutor
from .normal_symbolic_executor import NormalSymbolicExecutor
from .initial_symbolic_executor import InitialSymbolicExecutor
from ..opcode_translator.instruction_utils.instruction_utils import convert_instruction
from contextlib import contextmanager


class SymbolicTranslator:
    frame: types.FrameType
    _code_gen: PyCodeGen
    instructions: List[dis.Instruction]
    current_symbolic_frame_is_none: bool
  
    def __init__(self, frame: types.FrameType):
        self.frame = frame
        self._code_gen = PyCodeGen(frame.f_globals, frame.f_code)
        self.instructions = list(dis.get_instructions(self.frame.f_code))
        self.current_symbolic_frame_is_none = symbolic_frame_stack.top() is None

    def __call__(self) -> types.CodeType:
        self._code_gen_symbolic_executor_var()
        for i, instruction in enumerate(self.instructions):
            self._code_gen_try_add_pre_action(instruction_index=i)
            self._code_gen.add_pure_instructions([convert_instruction(instruction)])
            self._code_gen_try_add_post_action(instruction_index=i)
        return self._generate_code()

    def _generate_code(self):
        return self._code_gen.gen_pycode()

    def _code_gen_try_add_pre_action(self, instruction_index):
        instruction = self.instructions[instruction_index]
        opname = instruction.opname
        method_name = f"pre_{opname}"
        if hasattr(SymbolicExecutor, method_name):
            self._code_gen.load_fast(self.get_symbolic_executor_varname())
            self._code_gen.load_method("pre_action")
            self._code_gen.load_const(instruction_index)
            self._code_gen.call_method(argc=1)
            self._code_gen.pop_top()

    def _code_gen_try_add_post_action(self, instruction_index):
        instruction = self.instructions[instruction_index]
        opname = instruction.opname
        method_name = f"pre_{opname}"
        if not hasattr(SymbolicExecutor, method_name):
            self._code_gen.load_fast(self.get_symbolic_executor_varname())
            self._code_gen.load_const(instruction_index)
            self._code_gen.call_function(argc=1)
            self._code_gen.pop_top()

    def get_varname(self, prefix):
        return f"{prefix}_{str(id(self.frame.f_code))}"

    def _code_gen_symbolic_executor_var(self):
        symbolic_executor_type = None
        if symbolic_frame_stack.top() is None:
            symbolic_executor_type = InitialSymbolicExecutor
            SymbolicFrameMgr.create_initial_frame(self.frame)
        else:
            symbolic_executor_type = NormalSymbolicExecutor
        self._code_gen.load_global(
            symbolic_executor_type,
            self.get_varname(symbolic_executor_type.__name__)
        )
        self._code_gen.load_const(self.frame.f_code)
        self._code_gen.call_function(argc=1)
        self._code_gen.store_fast(self.get_symbolic_executor_varname())

    def get_symbolic_executor_varname(self):
        return self.get_varname(type(self).kExecutorNamePrefix)

    kExecutorNamePrefix = "symbolic_executor"
