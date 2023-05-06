from __future__ import annotations

import dis
import types
from typing import Callable, List, Tuple

from ...utils import (
    InnerError,
    Singleton,
    UnsupportError,
    is_strict_mode,
    log,
    log_do,
)
from ..instruction_utils import get_instructions
from .function_graph import FunctionGraph
from .tracker import ConstTracker, DummyTracker, GlobalTracker, LocalTracker
from .variables import (
    Guard,
    ListVariable,
    TupleVariable,
    VariableTracker,
    VariableTrackerFactory,
)

GuardedFunction = Tuple[types.CodeType, Guard]
GuardedFunctions = List[GuardedFunction]
CacheGetter = Callable[[types.FrameType, GuardedFunctions], types.CodeType]
dummy_guard: Guard = lambda frame: True


@Singleton
class InstructionTranslatorCache:
    cache: dict[types.CodeType, tuple[CacheGetter, GuardedFunctions]]

    def __init__(self):
        self.cache = {}

    def clear(self):
        self.cache.clear()

    def __call__(self, frame) -> types.CodeType:
        code: types.CodeType = frame.f_code
        if code not in self.cache:
            cache_getter, (new_code, guard_fn) = self.translate(frame)
            self.cache[code] = (cache_getter, [(new_code, guard_fn)])
            return new_code
        cache_getter, guarded_fns = self.cache[code]
        return cache_getter(frame, guarded_fns)

    def lookup(
        self, frame: types.FrameType, guarded_fns: GuardedFunctions
    ) -> types.CodeType:
        for code, guard_fn in guarded_fns:
            try:
                if guard_fn(frame):
                    log(3, "[Cache]: Cache hit\n")
                    return code
            except Exception as e:
                log(3, f"[Cache]: Guard function error: {e}\n")
                continue
        cache_getter, (new_code, guard_fn) = self.translate(frame)
        guarded_fns.append((new_code, guard_fn))
        return new_code

    def skip(
        self, frame: types.FrameType, guarded_fns: GuardedFunctions
    ) -> types.CodeType:
        log(3, f"[Cache]: Skip frame {frame.f_code.co_name}\n")
        return frame.f_code

    def translate(
        self, frame: types.FrameType
    ) -> tuple[CacheGetter, GuardedFunction]:
        code: types.CodeType = frame.f_code
        log(3, "[Cache]: Cache miss\n")
        result = start_translate(frame)
        if result is None:
            return self.skip, (code, dummy_guard)

        new_code, guard_fn = result
        return self.lookup, (new_code, guard_fn)


def start_translate(frame) -> GuardedFunction | None:
    simulator = OpcodeExecutor(frame)
    try:
        new_code, guard_fn = simulator.run()
        log_do(3, lambda: dis.dis(new_code))
        return new_code, guard_fn
    except InnerError as e:
        raise
    except UnsupportError as e:
        if is_strict_mode():
            raise
        log(2, f"Unsupport Frame is {frame.f_code.co_name}")
        return None
    except Exception as e:
        raise


class OpcodeExecutor:
    def __init__(self, frame: types.FrameType):
        self._frame = frame
        self._stack: list[VariableTracker] = []
        self._code = frame.f_code
        # fake env for run, new env should be gened by PyCodeGen
        self._co_consts = []
        self._locals = {}
        self._globals = {}
        self._lasti = 0  # idx of instruction list
        self.graph = FunctionGraph(self._frame)
        self.new_code = None

        self._instructions = get_instructions(self._code)
        self._prepare_virtual_env()

    def _prepare_virtual_env(self):
        for idx, (name, value) in enumerate(self._frame.f_locals.items()):
            name = self._frame.f_code.co_varnames[idx]
            self._locals[name] = VariableTrackerFactory.from_value(
                value, self.graph, LocalTracker(idx, name)
            )

        for name, value in self._frame.f_globals.items():
            self._globals[name] = VariableTrackerFactory.from_value(
                value, self.graph, GlobalTracker(name)
            )

        for value in self._code.co_consts:
            self._co_consts.append(
                VariableTrackerFactory.from_value(
                    value, self.graph, ConstTracker(value)
                )
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
        var = self._co_consts[instr.arg]
        self.push(var)

    def BINARY_MULTIPLY(self, instr):
        b = self.pop()
        a = self.pop()
        self.push(a * b)

    def BINARY_ADD(self, instr):
        b = self.pop()
        a = self.pop()
        self.push(a + b)

    def BINARY_SUBSCR(self, instr):
        b = self.pop()
        a = self.pop()
        self.push(a[b])

    def INPLACE_ADD(self, instr):
        b = self.pop()
        a = self.pop()
        a += b
        self.push(a)

    def CALL_METHOD(self, instr):
        TODO  # noqa: F821

    def RETURN_VALUE(self, instr):
        assert len(self._stack) == 1, "Stack must have one element."
        ret_val = self.pop()
        self.new_code, self.guard_fn = self.graph.start_compile(ret_val)

    def BUILD_LIST(self, instr):
        list_size = instr.arg
        if list_size <= len(self._stack):
            val_list = self._stack[-list_size:]
            self._stack[-list_size:] = []
            self.push(
                ListVariable(
                    val_list, graph=self.graph, tracker=DummyTracker(val_list)
                )
            )
        else:
            raise InnerError(
                f"OpExecutor want BUILD_LIST with size {list_size}, but current stack do not have enough elems."
            )

    def BUILD_TUPLE(self, instr):
        tuple_size = instr.arg
        if tuple_size <= len(self._stack):
            val_tuple = self._stack[-tuple_size:]
            self._stack[-tuple_size:] = []
            self.push(
                TupleVariable(
                    val_tuple, graph=self.graph, tracker=DummyTracker(val_tuple)
                )
            )
        else:
            raise InnerError(
                f"OpExecutor want BUILD_TUPLE with size {tuple_size}, but current stack do not have enough elems."
            )
