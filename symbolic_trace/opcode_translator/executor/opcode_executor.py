from __future__ import annotations

import copy
import dis
import inspect
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
    Instruction,
    get_instructions,
    modify_instrs,
    modify_vars,
)
from .flags import FORMAT_VALUE_FLAG as FV
from .function_graph import FunctionGraph
from .pycode_generator import (
    gen_code_options,
    gen_instr,
    gen_new_opcode,
    pycode_attributes,
)
from .tracker import DummyTracker, GetItemTracker, GlobalTracker, LocalTracker
from .variables import (
    ConstantVariable,
    ConstTracker,
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
    "==": lambda x, y: VariableTrackerFactory.from_value(
        x.value == y.value, None, tracker=DummyTracker([x, y])
    ),
}


class Stop:
    pass


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


def tos_op_warpper(fn):
    nargs = len(inspect.signature(fn).parameters)

    def inner(self: OpcodeExecutorBase, instr: Instruction):
        args = self.pop_n(nargs)
        self.push(fn(*args))

    return inner


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
        self.guard_fn = None
        self._prepare_virtual_env()

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
        return getattr(self, instr.opname)(instr)  # run single step.

    def indexof(self, instr):
        return self._instructions.index(instr)

    def create_ifelse_fn(self, index):
        instrs = copy.deepcopy(self._instructions)
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
                for var_name in self._frame.f_code.co_varnames
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

        return fn, fn_name, inputs

    def pop(self):
        return self._stack.pop()

    def pop_n(self, n):
        if n == 0:
            return []
        retval = self._stack[-n:]
        self._stack[-n:] = []
        return retval

    def push(self, val):
        self._stack.append(val)

    # unary operators
    # UNARY_POSITIVE = tos_op_warpper(operator.pos)
    UNARY_NEGATIVE = tos_op_warpper(operator.neg)
    # UNARY_NOT = tos_op_warpper(operator.not_)
    UNARY_INVERT = tos_op_warpper(operator.invert)

    # binary operators
    BINARY_POWER = tos_op_warpper(operator.pow)
    BINARY_MULTIPLY = tos_op_warpper(operator.mul)
    BINARY_MATRIX_MULTIPLY = tos_op_warpper(operator.matmul)
    BINARY_FLOOR_DIVIDE = tos_op_warpper(operator.floordiv)
    BINARY_TRUE_DIVIDE = tos_op_warpper(operator.truediv)
    BINARY_MODULO = tos_op_warpper(operator.mod)
    BINARY_ADD = tos_op_warpper(operator.add)
    BINARY_SUBTRACT = tos_op_warpper(operator.sub)
    # BINARY_LSHIFT = tos_op_warpper(operator.lshift)
    # BINARY_RSHIFT = tos_op_warpper(operator.rshift)
    BINARY_AND = tos_op_warpper(operator.and_)
    BINARY_OR = tos_op_warpper(operator.or_)
    BINARY_XOR = tos_op_warpper(operator.xor)

    # inplace operators
    # paddle variable do not have inplace operators. For example when call `y **= x`, will call var.__pow__
    INPLACE_POWER = tos_op_warpper(operator.ipow)
    INPLACE_MULTIPLY = tos_op_warpper(operator.imul)
    INPLACE_MATRIX_MULTIPLY = tos_op_warpper(operator.imatmul)
    INPLACE_FLOOR_DIVIDE = tos_op_warpper(operator.ifloordiv)
    INPLACE_TRUE_DIVIDE = tos_op_warpper(operator.itruediv)
    INPLACE_MODULO = tos_op_warpper(operator.imod)
    INPLACE_ADD = tos_op_warpper(operator.iadd)
    INPLACE_SUBTRACT = tos_op_warpper(operator.isub)
    # INPLACE_LSHIFT = tos_op_warpper(operator.ilshift)
    # INPLACE_RSHIFT = tos_op_warpper(operator.irshift)
    INPLACE_AND = tos_op_warpper(operator.iand)
    INPLACE_OR = tos_op_warpper(operator.ior)
    INPLACE_XOR = tos_op_warpper(operator.ixor)

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

    def BINARY_SUBSCR(self, instr):
        key = self.pop()
        container = self.pop()
        assert isinstance(key, VariableTracker)
        self._graph.add_global_guarded_variable(key)
        self.push(container[key.value])

    def STORE_SUBSCR(self, instr):
        key = self.pop()
        container = self.pop()
        value = self.pop()
        assert isinstance(key, VariableTracker)
        self._graph.add_global_guarded_variable(key)
        container[key.value] = value

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

    def CALL_FUNCTION_EX(self, instr):
        flag = instr.arg
        if flag & 0x01:  # has kwargs
            kwargs_variable = self.pop()
            assert isinstance(kwargs_variable, DictVariable)
            kwargs = kwargs_variable.unpack()
        else:
            kwargs = {}

        args_variable = self.pop()
        assert isinstance(args_variable, TupleVariable)
        args = args_variable.unpack()

        fn = self.pop()
        if isinstance(fn, FunctionVariable):
            ret = fn(*args, **kwargs)
            self.push(ret)

        else:
            raise UnsupportError(
                f"CALL_FUNCTION_EX: Currently only FunctionVariable are supported. meet type {type(fn)}"
            )

    def CALL_METHOD(self, instr):
        TODO  # noqa: F821

    def COMPARE_OP(self, instr):
        op = instr.argval
        if op in SUPPORT_COMPARE_OP:
            right, left = self.pop(), self.pop()
            self.push(SUPPORT_COMPARE_OP[op](left, right))
        else:
            raise UnsupportError()

    def POP_JUMP_IF_FALSE(self, instr):
        result = self.pop()
        if isinstance(result, TensorVariable):
            self._graph.start_compile(result)

            if_fn, if_fn_name, if_inputs = self.create_ifelse_fn(
                self.indexof(instr) + 1
            )
            self._graph.pycode_gen.gen_load_object(if_fn, if_fn_name)
            insert_index = len(self._graph.pycode_gen._instructions) - 1
            for name in if_inputs:
                self._graph.pycode_gen._add_instr("LOAD_FAST", argval=name)
            self._graph.pycode_gen.gen_call_function(
                argc=if_fn.__code__.co_argcount
            )
            self._graph.pycode_gen.gen_return()

            else_fn, else_fn_name, else_inputs = self.create_ifelse_fn(
                self.indexof(instr.jump_to)
            )
            self._graph.pycode_gen.gen_load_object(else_fn, else_fn_name)
            jump_to = self._graph.pycode_gen._instructions[-1]
            for name in else_inputs:
                self._graph.pycode_gen._add_instr("LOAD_FAST", argval=name)
            self._graph.pycode_gen.gen_call_function(
                argc=else_fn.__code__.co_argcount
            )
            self._graph.pycode_gen.gen_return()

            self._graph.pycode_gen._insert_instr(
                insert_index, "POP_JUMP_IF_FALSE", jump_to=jump_to
            )

            self.new_code = self._graph.pycode_gen.gen_pycode()
            self.guard_fn = self._graph.guard_fn
            return Stop()

    def JUMP_FORWARD(self, instr):
        self._lasti = self.indexof(instr.jump_to)

    def JUMP_ABSOLUTE(self, instr):
        self._lasti = self.indexof(instr.jump_to)

    def RETURN_VALUE(self, instr):
        assert len(self._stack) == 1, "Stack must have one element."
        ret_val = self.pop()
        self._graph.start_compile(ret_val)
        self._graph.pycode_gen.gen_return()
        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn
        return Stop()

    def BUILD_LIST(self, instr):
        list_size = instr.arg
        assert list_size <= len(
            self._stack
        ), f"OpExecutor want BUILD_LIST with size {list_size}, but current stack do not have enough elems."
        val_list = self.pop_n(list_size)
        self.push(
            VariableTrackerFactory.from_value(
                val_list, graph=self._graph, tracker=DummyTracker(val_list)
            )
        )

    def BUILD_TUPLE(self, instr):
        tuple_size = instr.arg
        assert tuple_size <= len(
            self._stack
        ), f"OpExecutor want BUILD_TUPLE with size {tuple_size}, but current stack do not have enough elems."
        val_tuple = self.pop_n(tuple_size)
        self.push(
            VariableTrackerFactory.from_value(
                tuple(val_tuple),
                graph=self._graph,
                tracker=DummyTracker(val_tuple),
            )
        )

    def BUILD_MAP(self, instr):
        map_size = instr.arg
        built_map = {}
        assert map_size * 2 <= len(
            self._stack
        ), f"OpExecutor want BUILD_MAP with size {map_size} * 2, but current stack do not have enough elems."
        val_for_dict = self.pop_n(map_size * 2)
        keys = val_for_dict[::2]
        values = val_for_dict[1::2]
        self.push(self.build_map(keys, values))

    def BUILD_CONST_KEY_MAP(self, instr):
        map_size = instr.arg
        assert map_size + 1 <= len(
            self._stack
        ), f"OpExecutor want BUILD_CONST_KEY_MAP with size {map_size} + 1, but current stack do not have enough elems."
        keys = self.pop().get_items()
        assert len(keys) == map_size
        values = self.pop_n(map_size)
        self.push(self.build_map(keys, values))

    def build_map(
        self, keys: list[VariableTracker], values: list[VariableTracker]
    ) -> VariableTracker:
        built_map = {}
        for key, value in zip(keys, values):
            assert isinstance(key, VariableTracker)
            # Add key to global guarded variable to avoid missing the key guard
            self._graph.add_global_guarded_variable(key)
            key = key.value
            built_map[key] = value
        return DictVariable(
            built_map,
            graph=self._graph,
            tracker=DummyTracker(keys + values),
        )

    def _rot_top_n(self, n):
        # a1 a2 a3 ... an  <- TOS
        # the stack changes to
        # an a1 a2 a3 an-1 <- TOS
        assert (
            len(self._stack) >= n
        ), f"There are not enough elements on the stack. {n} is needed."
        top = self.pop()
        self._stack[-(n - 1) : -(n - 1)] = [top]

    def POP_TOP(self, instr):
        self.pop()

    def ROT_TWO(self, instr):
        self._rot_top_n(2)

    def ROT_THREE(self, instr):
        self._rot_top_n(3)

    def ROT_FOUR(self, instr):
        self._rot_top_n(4)

    def UNPACK_SEQUENCE(self, instr):
        sequence = self.pop()

        '''
            TODO: To unpack iterator
            To unpack is easy, just like:
                seq = tuple(sequence.value)

            But what is the `source` when iterator returned a value ?
        '''
        if isinstance(sequence, TensorVariable):
            # TODO: If need to unpack a Tensor, should have different logic.
            raise NotImplementedError("Unpack a iterator is not implemented.")
        elif isinstance(sequence, (ListVariable, TupleVariable)):
            seq = sequence.value
        else:
            raise NotImplementedError(f"Unpack {sequence} is not implemented.")

        assert (
            len(seq) == instr.arg
        ), f"Want unpack {seq} to {instr.arg}, but the len is {len(seq)}."

        for i in range(instr.arg - 1, -1, -1):
            self.push(
                VariableTrackerFactory.from_value(
                    seq[i],
                    graph=self._graph,
                    tracker=GetItemTracker(sequence, i),
                )
            )

    def BUILD_STRING(self, instr):
        count = instr.arg
        assert count <= len(
            self._stack
        ), f"OpExecutor want BUILD_STRING with size {count}, but current stack do not have enough elems."
        str_list = self.pop_n(count)
        new_str = ''
        for s in str_list:
            assert isinstance(s.value, str)
            new_str += s.value
        self.push(ConstantVariable.wrap_literal(new_str))

    def FORMAT_VALUE(self, instr):

        flag = instr.arg
        which_conversion = flag & FV.FVC_MASK
        have_fmt_spec = bool((flag & FV.FVS_MASK) == FV.FVS_HAVE_SPEC)

        fmt_spec = self.pop().value if have_fmt_spec else ""
        value = self.pop()

        if which_conversion == FV.FVC_NONE:
            convert_fn = None
        elif which_conversion == FV.FVC_STR:
            convert_fn = "__str__"
        elif which_conversion == FV.FVC_REPR:
            convert_fn = "__repr__"
        elif which_conversion == FV.FVC_ASCII:
            convert_fn = "__ascii__"
        else:
            raise InnerError(
                f"Unexpected conversion flag {flag} for FORMAT_VALUE"
            )

        # different type will lead to different Tracker, so call self.push in different branch
        if isinstance(value, ConstantVariable):
            result = value.value
            if convert_fn is not None:
                result = getattr(result, convert_fn)(result)

            if not isinstance(result, str) or fmt_spec != "":
                result = format(result, fmt_spec)

            self.push(
                VariableTrackerFactory.from_value(
                    result, self._graph, DummyTracker([value])
                )
            )
        else:
            raise UnsupportError(f"Do not support format {type(value)} now")

    def build_seq_unpack(self, instr):
        oparg = instr.arg
        assert oparg <= len(self._stack)
        unpack_values = self.pop_n(oparg)

        retval = []
        for item in unpack_values:
            assert isinstance(item, (TupleVariable, ListVariable))
            retval.extend(item.unpack())

        if instr.opname in {
            "BUILD_TUPLE_UNPACK_WITH_CALL",
            "BUILD_TUPLE_UNPACK",
        }:
            retval = tuple(retval)

        self.push(
            VariableTrackerFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )

    def BUILD_TUPLE_UNPACK_WITH_CALL(self, instr):
        self.build_seq_unpack(instr)

    def BUILD_TUPLE_UNPACK(self, instr):
        self.build_seq_unpack(instr)

    def BUILD_LIST_UNPACK(self, instr):
        self.build_seq_unpack(instr)

    def BUILD_MAP_UNPACK(self, instr):
        oparg = instr.arg
        assert oparg <= len(self._stack)
        unpack_values = self.pop_n(oparg)

        retval = {}
        for item in unpack_values:
            assert isinstance(item.value, dict)
            retval.update(item.unpack())

        self.push(
            VariableTrackerFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )

    def BUILD_MAP_UNPACK_WITH_CALL(self, instr):
        oparg = instr.arg
        assert oparg <= len(self._stack)
        unpack_values = self.pop_n(oparg)

        retval = {}
        for item in unpack_values:
            assert isinstance(item.value, dict)
            wrapped_item = item.unpack()
            if wrapped_item.items() & retval.items():
                raise InnerError(
                    "BUILD_MAP_UNPACK_WITH_CALL found repeated key."
                )
            retval.update(wrapped_item)

        self.push(
            VariableTrackerFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )


class OpcodeExecutor(OpcodeExecutorBase):
    def __init__(self, frame):
        graph = FunctionGraph(frame)
        self._frame = frame
        super().__init__(frame.f_code, graph)

    def _prepare_virtual_env(self):
        for name, value in self._frame.f_locals.items():
            self._locals[name] = VariableTrackerFactory.from_value(
                value, self._graph, LocalTracker(name)
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
