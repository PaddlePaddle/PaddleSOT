from __future__ import annotations

import dis
import operator
import types
from typing import Callable, List, Tuple

from symbolic_trace.symbolic.bytecode_analysis import read_write_analysis
from symbolic_trace.utils.utils import generate_id

from ...utils import (
    InnerError,
    Singleton,
    UnsupportError,
    is_strict_mode,
    log,
    log_do,
)
from ..instruction_utils.instruction_utils import (
    get_instructions,
    modify_instrs,
    modify_vars,
)
from .function_graph import FunctionGraph
from .pycode_generator import (
    gen_code_options,
    gen_instr,
    gen_new_opcode,
    pycode_attributes,
)
from .tracker import (
    ConstTracker,
    DummyTracker,
    GetItemTracker,
    GlobalTracker,
    LocalTracker,
)
from .variables import (
    DictVariable,
    FunctionVariable,
    Guard,
    ListVariable,
    TensorVariable,
    TupleVariable,
    VariableTracker,
    VariableTrackerFactory,
)

GuardedFunction = Tuple[types.CodeType, Guard]
GuardedFunctions = List[GuardedFunction]
CacheGetter = Callable[[types.FrameType, GuardedFunctions], types.CodeType]
dummy_guard: Guard = lambda frame: True

SUPPORT_COMPARE_OP = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


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
        new_code, guard_fn = simulator.transform()
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


class OpcodeExecutorBase:
    def __init__(self, code: types.CodeType, graph: FunctionGraph):
        # fake env for run, new env should be gened by PyCodeGen
        self._stack = []
        self._co_consts = []
        self._locals = {}
        self._globals = {}
        self._lasti = 0  # idx of instruction list
        self._code = code
        self._instructions = get_instructions(self._code)
        self._graph = graph
        self.new_code = None
        self._prepare_virtual_env()

        self._code_options = gen_code_options(self._code)

    def _prepare_virtual_env(self):
        raise NotImplementedError("Please inplement virtual_env.")

    def transform(self):
        raise NotImplementedError()

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

    def step(self, instr):
        if not hasattr(self, instr.opname):
            raise UnsupportError(f"opcode: {instr.opname} is not supported.")
        log(3, f"[TraceExecution]: {instr.opname}, stack is {self._stack}\n")
        getattr(self, instr.opname)(instr)  # run single step.
        if (
            instr.opname == "RETURN_VALUE"
            or instr.opname == "POP_JUMP_IF_FALSE"
        ):
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
        self.push(self._globals[instr.argval])

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

    def BINARY_SUBTRACT(self, instr):
        b = self.pop()
        a = self.pop()
        self.push(a - b)

    def BINARY_SUBSCR(self, instr):
        b = self.pop()
        a = self.pop()
        self.push(a[b])

    def INPLACE_ADD(self, instr):
        b = self.pop()
        a = self.pop()
        a += b
        self.push(a)

    def CALL_FUNCTION(self, instr):
        args = []
        for _ in range(instr.argval):
            args.append(self.pop())
        args = args[::-1]
        fn = self.pop()
        if isinstance(fn, FunctionVariable):
            ret = fn(*args, {})
            self.push(ret)
        else:
            raise UnsupportError(
                f"CALL FUNCTION: Currently only FunctionVariable are supported. meet type {type(fn)}"
            )

    def CALL_METHOD(self, instr):
        TODO  # noqa: F821

    def COMPARE_OP(self, instr):
        op = instr.argval
        assert op in SUPPORT_COMPARE_OP
        right, left = self.pop(), self.pop()
        self.push(SUPPORT_COMPARE_OP[op](left, right))

    def POP_JUMP_IF_FALSE(self, instr):
        result = self.pop()
        if isinstance(result, TensorVariable):
            static_fn_code, guard_fn = self.graph.start_compile(
                result, if_return=False
            )
            self._code_options["co_names"].append(static_fn_code.co_name)

            if_instrs = self.create_ifelse_fn(self.indexof(instr) + 1)
            else_instrs = self.create_ifelse_fn(self.indexof(instr.jump_to))
            pop_jump_instr = gen_instr(
                'POP_JUMP_IF_FALSE', jump_to=else_instrs[0]
            )

            ret_instrs = (
                get_instructions(static_fn_code)
                + [pop_jump_instr]
                + if_instrs
                + else_instrs
            )

            modify_instrs(ret_instrs)
            modify_vars(ret_instrs, self._code_options)

            new_code = gen_new_opcode(
                ret_instrs, self._code_options, pycode_attributes
            )
            self.new_code = new_code
            self.guard_fn = guard_fn

    def JUMP_FORWARD(self, instr):
        self._lasti = self.indexof(instr.jump_to)

    def JUMP_ABSOLUTE(self, instr):
        self._lasti = self.indexof(instr.jump_to)

    def RETURN_VALUE(self, instr):
        assert len(self._stack) == 1, "Stack must have one element."
        ret_val = self.pop()
        self.new_code, self.guard_fn = self._graph.start_compile(ret_val)

    def BUILD_LIST(self, instr):
        list_size = instr.arg
        if list_size <= len(self._stack):
            val_list = self._stack[-list_size:]
            self._stack[-list_size:] = []
            self.push(
                ListVariable(
                    val_list, graph=self._graph, tracker=DummyTracker(val_list)
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
                    val_tuple,
                    graph=self._graph,
                    tracker=DummyTracker(val_tuple),
                )
            )
        else:
            raise InnerError(
                f"OpExecutor want BUILD_TUPLE with size {tuple_size}, but current stack do not have enough elems."
            )

    def BUILD_MAP(self, instr):
        map_size = instr.arg
        built_map = {}
        if map_size * 2 <= len(self._stack):
            val_for_dict = self._stack[-(map_size * 2) :]
            self._stack[-(map_size * 2) :] = []
            for i in range(map_size):
                key = val_for_dict[i]
                value = val_for_dict[i + 1]
                assert isinstance(key, VariableTracker)
                # Add key to global guarded variable to avoid missing the key guard
                self._graph.add_global_guarded_variable(key)
                key = key.value
                built_map[key] = value
            self.push(
                DictVariable(
                    built_map,
                    graph=self._graph,
                    tracker=DummyTracker(val_for_dict),
                )
            )
        else:
            raise InnerError(
                f"OpExecutor want BUILD_MAP with size {map_size} * 2, but current stack do not have enough elems."
            )

    def UNPACK_SEQUENCE(self, instr):
        sequence = self.pop()

        '''
            TODO: To unpack iterator
            To unpack is easy, just like:
                seq = tuple(sequence._sequence)

            But what is the `source` when iterator returned a value ?
        '''
        if isinstance(sequence, TensorVariable):
            # TODO: If need to unpack a Tensor, should have different logic.
            raise NotImplementedError("Unpack a iterator is not implemented.")
        elif isinstance(sequence, (ListVariable, TupleVariable)):
            seq = sequence._sequence
        else:
            raise NotImplementedError(f"Unpack {sequence} is not implemented.")

        if len(seq) != instr.arg:
            raise InnerError(
                f"Want unpack {seq} to {instr.arg}, but the len is {len(seq)}."
            )

        for i in range(instr.arg - 1, -1, -1):
            if not isinstance(seq[i], VariableTracker):
                self.push(
                    VariableTrackerFactory.from_value(
                        seq[i],
                        self._graph,
                        GetItemTracker(
                            sequence,
                            VariableTrackerFactory.from_value(
                                i, self._graph, ConstTracker(i)
                            ),
                        ),
                    )
                )
            else:
                self.push(seq[i])

    def indexof(self, instr):
        return self._instructions.index(instr)

    def create_ifelse_fn(self, index):
        instrs = get_instructions(self._code)
        inputs = read_write_analysis(instrs, index)
        instrs = [gen_instr('JUMP_ABSOLUTE', jump_to=instrs[index])] + instrs

        fn_code_options = gen_code_options(self._code)
        # update fn_code_options
        fn_code_options['co_argcount'] = len(inputs)
        # inputs shold be at the front of the co_varnames
        fn_code_options['co_varnames'] = tuple(
            list(inputs)
            + [
                var_name
                for var_name in self._code_options['co_varnames']
                if var_name not in inputs
            ]
        )

        modify_instrs(instrs)
        modify_vars(instrs, fn_code_options)

        new_code = gen_new_opcode(instrs, fn_code_options, pycode_attributes)

        fn_name = 'ifelse_fn_at_{}_{}'.format(
            instrs[index].offset, generate_id()
        )
        fn = types.FunctionType(new_code, self._globals, fn_name)

        # add sub function to frame.f_global
        self._frame.f_globals[fn_name] = fn
        self._code_options["co_names"].append(fn_name)

        ret_instrs = []
        idx = len(self._code_options["co_names"]) - 1
        ret_instrs.append(gen_instr("LOAD_GLOBAL", arg=idx, argval=fn_name))
        for name in inputs:
            ret_instrs.append(gen_instr("LOAD_FAST", argval=name))
        ret_instrs.append(
            gen_instr(
                "CALL_FUNCTION",
                arg=new_code.co_argcount,
                argval=new_code.co_argcount,
            )
        )
        ret_instrs.append(gen_instr("RETURN_VALUE"))

        return ret_instrs


class OpcodeExecutor(OpcodeExecutorBase):
    def __init__(self, frame):
        graph = FunctionGraph(frame)
        self._frame = frame
        super().__init__(frame.f_code, graph)

    def _prepare_virtual_env(self):
        for idx, (name, value) in enumerate(self._frame.f_locals.items()):
            name = self._frame.f_code.co_varnames[idx]
            self._locals[name] = VariableTrackerFactory.from_value(
                value, self._graph, LocalTracker(idx, name)
            )

        for name, value in self._frame.f_globals.items():
            self._globals[name] = VariableTrackerFactory.from_value(
                value, self._graph, GlobalTracker(name)
            )

        for value in self._code.co_consts:
            self._co_consts.append(
                VariableTrackerFactory.from_value(
                    value, self._graph, ConstTracker(value)
                )
            )

    def transform(self):
        self.run()
        if self.new_code is None:
            raise InnerError("OpExecutor return a empty new_code.")
        return self.new_code, self.guard_fn
