import dis
import types

from ..utils import InnerError, UnsupportError, log
from .function_graph import FunctionGraph
from .source import *
from .variables import *


class OpcodeExecutor:
    def __init__(self, frame: types.FrameType, code_options):
        self._code_options = code_options
        self._frame = frame
        self._stack = []
        self._code = frame.f_code
        self._co_consts = self._code.co_consts
        self._locals = {}
        self._globals = {}
        self._lasti = 0  # idx of instruction list
        self.graph = FunctionGraph(self._frame)
        self.new_code = None

        # Instructions is struture like the following:
        # Instruction(opname='LOAD_CONST',
        #             opcode=100, arg=1, argval=2, argrepr='2', offset=0,
        #             starts_line=11, is_jump_target=False)
        self._instructions = list(dis.get_instructions(self._code))
        # offset -> instruction
        self.offset_map = {}
        self._prepare_locals_and_globals()

    def _prepare_locals_and_globals(self):
        for name, value in self._frame.f_locals.items():
            self._locals[name] = VariableTrackerFactory.from_value(
                value, self.graph
            )

        for name, value in self._frame.f_globals.items():
            self._globals[name] = VariableTrackerFactory.from_value(
                value, self.graph
            )

    def run(self):
        log(3, f"start execute opcode: {self._code}\n")
        self._lasti = 0
        while True:
            if self._lasti >= len(self._instructions):
                raise InnerError("lasti out of range, InnerError.")
            cur_instr = self._instructions[self._lasti]
            self._lasti += 1
            is_stop = self.step(cur_instr)
            if is_stop:
                break
        if self.new_code is None:
            raise InnerError("OpExecutor return a emtpy new_code.")
        return self.new_code, self.guard_fn

    def step(self, instr):
        if not hasattr(self, instr.opname):
            raise UnsupportError(f"opcode: {instr.opname} is not supported.")
        log(3, f"[TraceExecution]: {instr.opname}, stack is {self._stack}\n")
        getattr(self, instr.opname)(instr)  # run single step.
        if instr.opname == "RETURN_VALUE":
            return True
        return False

    def pop(self):
        return self._stack.pop()

    def push(self, val):
        self._stack.append(val)

    def LOAD_ATTR(self, instr):
        pass

    def LOAD_FAST(self, instr):
        varname = instr.argval
        var = self._locals[varname]
        var = VariableTrackerFactory.from_value(var, self.graph)
        var.set_source(LocalSource(instr.arg, varname))
        self.push(var)

    def LOAD_METHOD(self, instr):
        pass

    def BINARY_ADD(self, instr):
        pass

    def STORE_FAST(self, instr):
        """
        TODO: side effect may happen
        """
        var = self.pop()
        self._locals[instr.argval] = var

    def LOAD_GLOBAL(self, instr):
        pass

    def LOAD_CONST(self, instr):
        var = ConstantVariable(instr.argval)
        self.push(var)

    def BINARY_MULTIPLY(self, instr):
        b = self.pop()
        a = self.pop()
        self.push(a * b)

    def BINARY_ADD(self, instr):
        b = self.pop()
        a = self.pop()
        self.push(a + b)

    def CALL_METHOD(self, instr):
        pass

    def RETURN_VALUE(self, instr):
        assert len(self._stack) == 1, "Stack must have one element."
        ret_val = self.pop()
        self.new_code, self.guard_fn = self.graph.start_compile(ret_val)
