import dis
import types

from ...utils import (
    Cache,
    InnerError,
    Singleton,
    UnsupportError,
    freeze_structure,
    log,
    log_do,
)
from ..instruction_utils import get_instructions
from .function_graph import FunctionGraph
from .source import LocalSource
from .variables import ConstantVariable, VariableTrackerFactory


@Singleton
class InstructionTranslatorCache(Cache):
    def key_fn(self, *args, **kwargs):
        code, *others = args
        return freeze_structure((code))

    def value_fn(self, *args, **kwargs):
        return start_translate(*args, **kwargs)


def start_translate(frame):
    simulator = OpcodeExecutor(frame)
    try:
        new_code, guard_fn = simulator.run()
        log_do(3, lambda: dis.dis(new_code))
        return new_code
    except InnerError as e:
        raise
    except UnsupportError as e:
        log(2, f"Unsupport Frame is {frame.f_code.co_name}")
        return frame.f_code
    except Exception as e:
        raise


class OpcodeExecutor:
    def __init__(self, frame: types.FrameType):
        self._frame = frame
        self._stack = []
        self._code = frame.f_code
        self._co_consts = self._code.co_consts
        self._locals = {}
        self._globals = {}
        self._lasti = 0  # idx of instruction list
        self.graph = FunctionGraph(self._frame)
        self.new_code = None

        self._instructions = get_instructions(self._code)
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
        TODO  # noqa: F821

    def LOAD_FAST(self, instr):
        varname = instr.argval
        var = self._locals[varname]
        var = VariableTrackerFactory.from_value(var, self.graph)
        var.set_source(LocalSource(instr.arg, varname))
        self.push(var)

    def LOAD_METHOD(self, instr):
        TODO  # noqa: F821

    def STORE_FAST(self, instr):
        """
        TODO: side effect may happen
        """
        var = self.pop()
        self._locals[instr.argval] = var

    def LOAD_GLOBAL(self, instr):
        TODO  # noqa: F821

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
        TODO  # noqa: F821

    def RETURN_VALUE(self, instr):
        assert len(self._stack) == 1, "Stack must have one element."
        ret_val = self.pop()
        self.new_code, self.guard_fn = self.graph.start_compile(ret_val)
