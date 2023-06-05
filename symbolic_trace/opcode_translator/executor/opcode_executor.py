from __future__ import annotations

import collections
import dis
import functools
import inspect
import operator
import types
from typing import Callable, List, Optional, Tuple

from ...utils import (
    BreakGraphError,
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
    instrs_info,
)
from ..instruction_utils.opcode_analysis import analysis_inputs
from .function_graph import FunctionGraph
from .guard import Guard
from .instr_flag import FORMAT_VALUE_FLAG as FV
from .instr_flag import MAKE_FUNCTION_FLAG as MF
from .pycode_generator import PyCodeGen
from .tracker import (
    BuiltinTracker,
    DummyTracker,
    GetItemTracker,
    GetIterTracker,
    GlobalTracker,
    LocalTracker,
)
from .variables import (
    CallableVariable,
    ConstantVariable,
    ConstTracker,
    ContainerVariable,
    DictIterVariable,
    DictVariable,
    DummyVariable,
    IterVariable,
    ListVariable,
    SequenceIterVariable,
    TensorIterVariable,
    TensorVariable,
    TupleVariable,
    UserDefinedFunctionVariable,
    UserDefinedIterVariable,
    VariableBase,
    VariableFactory,
)

CustomCode = collections.namedtuple(
    "CustomCode", ["code", "disable_eval_frame"]
)


GuardedFunction = Tuple[types.CodeType, Guard]
GuardedFunctions = List[GuardedFunction]
CacheGetter = Callable[
    [types.FrameType, GuardedFunctions], Optional[CustomCode]
]
dummy_guard: Guard = lambda frame: True

SUPPORT_COMPARE_OP = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": lambda x, y: VariableFactory.from_value(
        x.value == y.value, None, tracker=DummyTracker([x, y])
    ),
    "!=": lambda x, y: VariableFactory.from_value(
        x.value != y.value, None, tracker=DummyTracker([x, y])
    ),
    "is not": lambda x, y: VariableFactory.from_value(
        x.value is not y.value, None, tracker=DummyTracker([x, y])
    ),
    "is": lambda x, y: VariableFactory.from_value(
        x.value is y.value, None, tracker=DummyTracker([x, y])
    ),
}


class Stop:
    pass


@Singleton
class InstructionTranslatorCache:
    cache: dict[types.CodeType, tuple[CacheGetter, GuardedFunctions]]
    translate_count: int

    def __init__(self):
        self.cache = {}
        self.translate_count = 0

    def clear(self):
        self.cache.clear()
        self.translate_count = 0

    def __call__(self, frame) -> CustomCode | None:
        code: types.CodeType = frame.f_code
        if code not in self.cache:
            cache_getter, (new_code, guard_fn) = self.translate(frame)
            self.cache[code] = (cache_getter, [(new_code, guard_fn)])
            if cache_getter == self.skip:
                return None
            return CustomCode(new_code, False)
        cache_getter, guarded_fns = self.cache[code]
        return cache_getter(frame, guarded_fns)

    def lookup(
        self, frame: types.FrameType, guarded_fns: GuardedFunctions
    ) -> CustomCode | None:
        for code, guard_fn in guarded_fns:
            try:
                if guard_fn(frame):
                    log(3, "[Cache]: Cache hit\n")
                    return CustomCode(code, True)
            except Exception as e:
                log(3, f"[Cache]: Guard function error: {e}\n")
                continue
        cache_getter, (new_code, guard_fn) = self.translate(frame)
        guarded_fns.append((new_code, guard_fn))
        return CustomCode(new_code, False)

    def skip(
        self, frame: types.FrameType, guarded_fns: GuardedFunctions
    ) -> CustomCode | None:
        log(3, f"[Cache]: Skip frame {frame.f_code.co_name}\n")
        return None

    def translate(
        self, frame: types.FrameType
    ) -> tuple[CacheGetter, GuardedFunction]:
        code: types.CodeType = frame.f_code
        log(3, "[Cache]: Cache miss\n")
        self.translate_count += 1

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
    # TODO(0x45f): handle BreakGraphError to trigger fallback
    except (UnsupportError, BreakGraphError) as e:
        if is_strict_mode():
            raise
        log(
            2,
            f"Unsupport Frame is {frame.f_code}, error message is: {str(e)}\n",
        )
        return None
    except Exception as e:
        raise


def tos_op_wrapper(fn):
    nargs = len(inspect.signature(fn).parameters)

    def inner(self: OpcodeExecutorBase, instr: Instruction):
        args = self.pop_n(nargs)
        self.push(fn(*args))

    return inner


def breakoff_graph_with_jump(normal_jump):
    """breakoff graph when meet jump."""

    def jump_instruction_with_fallback(self: OpcodeExecutor, instr):
        result = self.peek()
        if isinstance(result, TensorVariable):
            self.pop()
            # fallback when in OpcodeExecutor
            # raise error in OpcodeInlineExecutor
            self._fallback_in_jump(result, instr)
            return Stop()
        else:
            return normal_jump(self, instr)

    return jump_instruction_with_fallback


def break_graph_in_call(push_n):
    def decorate(call_fn):
        @functools.wraps(call_fn)
        def wrapper(self: OpcodeExecutor, instr):
            origin_stack = list(self._stack)
            try:
                return call_fn(self, instr)
            except BreakGraphError as e:
                index = self.indexof(instr)
                self._stack = origin_stack

                # gen call static fn opcode
                ret_vars = [
                    arg
                    for arg in self._stack
                    if isinstance(arg, TensorVariable)
                ]
                resume_input_name = analysis_inputs(
                    self._instructions, index + 1
                )
                ret_vars = ret_vars + [
                    self.get_var(name)
                    for name in resume_input_name
                    if self.get_var(name) not in ret_vars
                ]
                self._graph.start_compile(*ret_vars)
                for _ in ret_vars:
                    self._graph.pycode_gen.gen_pop_top()

                # gen graph break call fn opcode
                for stack_arg in self._stack:
                    stack_arg.reconstruct(self._graph.pycode_gen)
                self._graph.pycode_gen.add_pure_instructions([instr])

                # gen call resume fn opcode
                stack_effect = dis.stack_effect(instr.opcode, instr.arg)
                self.pop_n(push_n - stack_effect)
                stack_size = len(self._stack) + push_n
                self._graph.pycode_gen.gen_build_tuple(stack_size)
                resume_fn, _ = self._create_resume_fn(index + 1, stack_size)
                self._graph.pycode_gen.gen_load_object(
                    resume_fn, resume_fn.__code__.co_name
                )
                self._graph.pycode_gen._add_instr('ROT_TWO')
                self._graph.pycode_gen.gen_unpack_sequence(stack_size)
                for name in resume_input_name:
                    self._locals[name].reconstruct(self._graph.pycode_gen)
                self._graph.pycode_gen.gen_call_function(
                    argc=resume_fn.__code__.co_argcount
                )

                # gen RETURN_VALUE
                self._graph.pycode_gen.gen_return()

                self.new_code = self._graph.pycode_gen.gen_pycode()
                self.guard_fn = self._graph.guard_fn

                return Stop()

        return wrapper

    return decorate


class OpcodeExecutorBase:
    def __init__(self, code: types.CodeType, graph: FunctionGraph):
        # fake env for run, new env should be gened by PyCodeGen
        self._stack: list[VariableBase] = []
        self._co_consts = []
        self._locals = {}
        self._globals = {}
        self._builtins = {}
        self._lasti = 0  # idx of instruction list
        self._code = code
        self._instructions = get_instructions(self._code)
        self._graph = graph
        self.new_code = None
        self.guard_fn = None
        self._prepare_virtual_env()

    def print_instrs(self):
        print(instrs_info(self._instructions))

    def print_sir(self):
        print(self._graph.sir_ctx.TOS)

    def _prepare_virtual_env(self):
        raise NotImplementedError("Please inplement virtual_env.")

    def _fallback_in_jump(self, result, instr):
        raise NotImplementedError()

    def transform(self):
        raise NotImplementedError()

    def get_var(self, name: str):
        if name in self._locals.keys():
            return self._locals[name]
        elif name in self._globals.keys():
            return self._globals[name]
        elif name in self._builtins.keys():
            return self._builtins[name]
        else:
            raise InnerError(f'Can not get var: {name}')

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

    def pop(self) -> VariableBase:
        return self._stack.pop()

    def peek(self) -> VariableBase:
        return self._stack[-1]

    def peek_n(self, n) -> list[VariableBase]:
        return self._stack[-n:]

    def pop_n(self, n: int) -> list[VariableBase]:
        if n == 0:
            return []
        retval = self._stack[-n:]
        self._stack[-n:] = []
        return retval

    def push(self, val: VariableBase):
        self._stack.append(val)

    def DUP_TOP(self, instr):
        self.push(self.peek())

    def DUP_TOP_TWO(self, instr):
        for ref in self.peek_n(2):
            self.push(ref)

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

    # unary operators
    UNARY_POSITIVE = tos_op_wrapper(operator.pos)
    UNARY_NEGATIVE = tos_op_wrapper(operator.neg)
    # UNARY_NOT = tos_op_wrapper(operator.not_)
    UNARY_INVERT = tos_op_wrapper(operator.invert)

    # binary operators
    BINARY_POWER = tos_op_wrapper(operator.pow)
    BINARY_MULTIPLY = tos_op_wrapper(operator.mul)
    BINARY_MATRIX_MULTIPLY = tos_op_wrapper(operator.matmul)
    BINARY_FLOOR_DIVIDE = tos_op_wrapper(operator.floordiv)
    BINARY_TRUE_DIVIDE = tos_op_wrapper(operator.truediv)
    BINARY_MODULO = tos_op_wrapper(operator.mod)
    BINARY_ADD = tos_op_wrapper(operator.add)
    BINARY_SUBTRACT = tos_op_wrapper(operator.sub)
    BINARY_LSHIFT = tos_op_wrapper(operator.lshift)
    BINARY_RSHIFT = tos_op_wrapper(operator.rshift)
    BINARY_AND = tos_op_wrapper(operator.and_)
    BINARY_OR = tos_op_wrapper(operator.or_)
    BINARY_XOR = tos_op_wrapper(operator.xor)

    def BINARY_SUBSCR(self, instr):
        key = self.pop()
        container = self.pop()
        assert isinstance(key, VariableBase)
        self._graph.add_global_guarded_variable(key)
        self.push(container[key.value])

    # inplace operators
    # paddle variable do not have inplace operators. For example when call `y **= x`, will call var.__pow__
    INPLACE_POWER = tos_op_wrapper(operator.ipow)
    INPLACE_MULTIPLY = tos_op_wrapper(operator.imul)
    INPLACE_MATRIX_MULTIPLY = tos_op_wrapper(operator.imatmul)
    INPLACE_FLOOR_DIVIDE = tos_op_wrapper(operator.ifloordiv)
    INPLACE_TRUE_DIVIDE = tos_op_wrapper(operator.itruediv)
    INPLACE_MODULO = tos_op_wrapper(operator.imod)
    INPLACE_ADD = tos_op_wrapper(operator.iadd)
    INPLACE_SUBTRACT = tos_op_wrapper(operator.isub)
    INPLACE_LSHIFT = tos_op_wrapper(operator.ilshift)
    INPLACE_RSHIFT = tos_op_wrapper(operator.irshift)
    INPLACE_AND = tos_op_wrapper(operator.iand)
    INPLACE_OR = tos_op_wrapper(operator.ior)
    INPLACE_XOR = tos_op_wrapper(operator.ixor)

    def NOP(self, instr):
        pass

    def LOAD_ATTR(self, instr):
        attr_name = instr.argval
        obj = self.pop()
        self.push(getattr(obj, attr_name))

    def LOAD_CONST(self, instr):
        var = self._co_consts[instr.arg]
        self.push(var)

    def LOAD_FAST(self, instr):
        varname = instr.argval
        var = self._locals[varname]
        self.push(var)

    def LOAD_GLOBAL(self, instr):
        name = instr.argval
        if name in self._globals.keys():
            value = self._globals[name]
        else:
            value = self._builtins[name]
        self.push(value)

    def LOAD_METHOD(self, instr):
        method_name = instr.argval
        obj = self.pop()
        method = getattr(obj, method_name)
        self.push(method)
        self.push(DummyVariable())

    def STORE_FAST(self, instr):
        """
        TODO: side effect may happen
        """
        var = self.pop()
        self._locals[instr.argval] = var

    def STORE_SUBSCR(self, instr):
        key = self.pop()
        container = self.pop()
        value = self.pop()
        assert isinstance(key, VariableBase)
        self._graph.add_global_guarded_variable(key)
        container[key.value] = value

    def BUILD_LIST(self, instr):
        list_size = instr.arg
        assert list_size <= len(
            self._stack
        ), f"OpExecutor want BUILD_LIST with size {list_size}, but current stack do not have enough elems."
        val_list = self.pop_n(list_size)
        self.push(
            VariableFactory.from_value(
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
            VariableFactory.from_value(
                tuple(val_tuple),
                graph=self._graph,
                tracker=DummyTracker(val_tuple),
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

    def BUILD_SLICE(self, instr):
        if instr.arg == 3:
            step = self.pop()
        else:
            step = None
        stop = self.pop()
        start = self.pop()

        related_list = [start, stop, step] if step else [start, stop]

        slice_ = slice(*(x.value for x in related_list))

        self.push(
            VariableFactory.from_value(
                slice_, self._graph, DummyTracker(related_list)
            )
        )

    def build_map(
        self, keys: list[VariableBase], values: list[VariableBase]
    ) -> VariableBase:
        built_map = {}
        for key, value in zip(keys, values):
            assert isinstance(key, VariableBase)
            # Add key to global guarded variable to avoid missing the key guard
            self._graph.add_global_guarded_variable(key)
            key = key.value
            built_map[key] = value
        return DictVariable(
            built_map,
            graph=self._graph,
            tracker=DummyTracker(keys + values),
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

    def build_seq_unpack(self, instr):
        oparg = instr.arg
        assert oparg <= len(self._stack)
        unpack_values = self.pop_n(oparg)

        retval = []
        for item in unpack_values:
            assert isinstance(item, (TupleVariable, ListVariable))
            retval.extend(item.get_wrapped_items())

        if instr.opname in {
            "BUILD_TUPLE_UNPACK_WITH_CALL",
            "BUILD_TUPLE_UNPACK",
        }:
            retval = tuple(retval)

        self.push(
            VariableFactory.from_value(
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
            retval.update(item.get_wrapped_items())

        self.push(
            VariableFactory.from_value(
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
            wrapped_item = item.get_wrapped_items()
            if wrapped_item.items() & retval.items():
                raise InnerError(
                    "BUILD_MAP_UNPACK_WITH_CALL found repeated key."
                )
            retval.update(wrapped_item)

        self.push(
            VariableFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )

    @break_graph_in_call(push_n=1)
    def CALL_FUNCTION(self, instr):
        n_args = instr.arg
        assert n_args <= len(self._stack)
        args = self.pop_n(n_args)
        kwargs = {}
        fn = self.pop()
        if not isinstance(fn, CallableVariable):
            raise UnsupportError(f"CALL_FUNCTION: {fn} is not callable")
        ret = fn(*args, **kwargs)
        self.push(ret)

    def CALL_FUNCTION_KW(self, instr):
        n_args = instr.arg
        assert n_args + 2 <= len(self._stack)

        kwargs_keys = self.pop()
        assert isinstance(kwargs_keys, TupleVariable)
        assert len(kwargs_keys) > 0
        kwargs_keys = [
            x.value if isinstance(x, VariableBase) else x
            for x in kwargs_keys.value
        ]

        # split arg_list to args and kwargs
        arg_list = self.pop_n(n_args)
        args = arg_list[0 : -len(kwargs_keys)]
        kwargs_values = arg_list[-len(kwargs_keys) :]
        kwargs = dict(zip(kwargs_keys, kwargs_values))

        fn = self.pop()
        if not isinstance(fn, CallableVariable):
            raise UnsupportError(f"CALL_FUNCTION_KW: {fn} is not callable.")
        ret = fn(*args, **kwargs)
        self.push(ret)

    def CALL_FUNCTION_EX(self, instr):
        flag = instr.arg
        if flag & 0x01:  # has kwargs
            kwargs_variable = self.pop()
            assert isinstance(kwargs_variable, DictVariable)
            kwargs = kwargs_variable.get_wrapped_items()
        else:
            kwargs = {}

        args_variable = self.pop()
        assert isinstance(args_variable, TupleVariable)
        args = args_variable.get_wrapped_items()

        fn = self.pop()
        if not isinstance(fn, CallableVariable):
            raise UnsupportError(f"CALL_FUNCTION_EX: {fn} is not callable.")
        ret = fn(*args, **kwargs)
        self.push(ret)

    @break_graph_in_call(push_n=1)
    def CALL_METHOD(self, instr):
        n_args = instr.argval
        assert n_args <= len(self._stack)
        args = self.pop_n(n_args)
        assert isinstance(self.pop(), DummyVariable)
        method = self.pop()
        if not isinstance(method, CallableVariable):
            raise UnsupportError(f"CALL METHOD: {method} is not callable.")
        ret = method(*args)
        self.push(ret)

    def COMPARE_OP(self, instr):
        op = instr.argval
        if op in SUPPORT_COMPARE_OP:
            right, left = self.pop(), self.pop()
            self.push(SUPPORT_COMPARE_OP[op](left, right))
            return
        else:
            raise UnsupportError(
                f"{instr} is not support. may be not a supported compare op."
            )

    def MAKE_FUNCTION(self, instr):
        fn_name = self.pop()
        codeobj = self.pop()
        global_dict = self._globals

        related_list = [fn_name, codeobj]

        flag = instr.arg
        if flag & MF.MF_HAS_CLOSURE:
            # closure should be a tuple of Variables
            closure_variable = self.pop()
            assert isinstance(closure_variable, TupleVariable)
            related_list.append(closure_variable)
            closure = tuple(closure_variable.get_wrapped_items())
        else:
            closure = ()

        if flag & MF.MF_HAS_ANNOTATION:
            # can not set annotation in python env, skip it
            related_list.append(self.pop())

        if flag & MF.MF_HAS_KWDEFAULTS:
            raise UnsupportError(
                "Found need func_kwdefaults when MAKE_FUNCTION."
            )

        if flag & MF.MF_HAS_DEFAULTS:
            '''
            default_args should have tracker too, like:

            def f(x):
                def g(z=x):
                    pass
            '''
            default_args_variable = self.pop()
            assert isinstance(default_args_variable, TupleVariable)
            related_list.append(default_args_variable)
            default_args = tuple(default_args_variable.get_wrapped_items())
        else:
            default_args = ()

        new_fn = types.FunctionType(
            codeobj.value, global_dict, fn_name.value, default_args, closure
        )

        self.push(
            UserDefinedFunctionVariable(
                new_fn, self._graph, DummyTracker(related_list)
            )
        )

    def GET_ITER(self, instr):
        source_obj = self.pop()
        if isinstance(source_obj, IterVariable):
            return self.push(source_obj)

        if isinstance(source_obj, (ListVariable, TupleVariable)):
            self.push(
                SequenceIterVariable(
                    source_obj, self._graph, GetIterTracker(source_obj)
                )
            )
        elif isinstance(source_obj, DictVariable):
            self.push(
                DictIterVariable(
                    source_obj, self._graph, GetIterTracker(source_obj)
                )
            )
        elif isinstance(source_obj, TensorVariable):
            self.push(
                TensorIterVariable(
                    source_obj, self._graph, GetIterTracker(source_obj)
                )
            )
        else:
            self.push(
                UserDefinedIterVariable(
                    source_obj, self._graph, GetIterTracker(source_obj)
                )
            )

    def FOR_ITER(self, instr):
        iterator = self.pop()
        assert isinstance(iterator, IterVariable)

        # simplely get next
        if isinstance(iterator, (SequenceIterVariable, DictIterVariable)):
            try:
                val, next_iterator = iterator.next()
                self.push(
                    next_iterator
                )  # need a new iterator to replace the old one
                self.push(val)
            except StopIteration:
                self._lasti = self.indexof(instr.jump_to)

        # TODO need support TensorIterVariable.next

        else:
            self._fallback_in_for_loop(iterator, instr)
            return Stop()

    def JUMP_FORWARD(self, instr):
        self._lasti = self.indexof(instr.jump_to)

    def JUMP_ABSOLUTE(self, instr):
        self._lasti = self.indexof(instr.jump_to)

    @breakoff_graph_with_jump
    def JUMP_IF_FALSE_OR_POP(self, instr):
        pred_obj = self.peek()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = not bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            else:
                self.pop()
            return
        raise UnsupportError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @breakoff_graph_with_jump
    def JUMP_IF_TRUE_OR_POP(self, instr):
        pred_obj = self.peek()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            else:
                self.pop()
            return
        raise UnsupportError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @breakoff_graph_with_jump
    def POP_JUMP_IF_FALSE(self, instr):
        pred_obj = self.pop()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = not bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            return
        raise UnsupportError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @breakoff_graph_with_jump
    def POP_JUMP_IF_TRUE(self, instr):
        pred_obj = self.pop()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            return
        raise UnsupportError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

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
                VariableFactory.from_value(
                    seq[i],
                    graph=self._graph,
                    tracker=GetItemTracker(sequence, i),
                )
            )

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
                VariableFactory.from_value(
                    result, self._graph, DummyTracker([value])
                )
            )
        else:
            raise UnsupportError(f"Do not support format {type(value)} now")

    def RETURN_VALUE(self, instr):
        assert (
            len(self._stack) == 1
        ), f"Stack must have one element, but get {len(self._stack)} elements."
        ret_val = self.pop()
        self._graph.start_compile(ret_val)
        self._graph.pycode_gen.gen_return()
        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn
        return Stop()


class OpcodeExecutor(OpcodeExecutorBase):
    def __init__(self, frame):
        graph = FunctionGraph(frame)
        self._frame = frame
        super().__init__(frame.f_code, graph)

    def _prepare_virtual_env(self):
        for name, value in self._frame.f_locals.items():
            self._locals[name] = VariableFactory.from_value(
                value, self._graph, LocalTracker(name)
            )

        for name, value in self._frame.f_globals.items():
            self._globals[name] = VariableFactory.from_value(
                value, self._graph, GlobalTracker(name)
            )

        for name, value in self._frame.f_builtins.items():
            self._builtins[name] = VariableFactory.from_value(
                value, self._graph, BuiltinTracker(name)
            )

        for value in self._code.co_consts:
            self._co_consts.append(
                VariableFactory.from_value(
                    value, self._graph, ConstTracker(value)
                )
            )

    def _create_resume_fn(self, index, stack_size=0):
        pycode_gen = PyCodeGen(self._frame)
        fn, inputs = pycode_gen.gen_resume_fn_at(index, stack_size)
        return fn, inputs

    def _fallback_in_jump(self, result, instr):
        if_fn, if_inputs = self._create_resume_fn(self.indexof(instr) + 1)
        else_fn, else_inputs = self._create_resume_fn(
            self.indexof(instr.jump_to)
        )
        inputs_name = if_inputs | else_inputs
        inputs_var = [
            self.get_var(name)
            for name in inputs_name
            if self.get_var(name) is not result
        ]
        ret_vars = [
            result,
        ] + inputs_var
        self._graph.start_compile(*ret_vars)
        for _ in inputs_var:
            self._graph.pycode_gen.gen_pop_top()

        if if_fn is not None:
            self._graph.pycode_gen.gen_load_object(
                if_fn, if_fn.__code__.co_name
            )
            insert_index = len(self._graph.pycode_gen._instructions) - 1
            for name in if_inputs:
                self.get_var(name).reconstruct(self._graph.pycode_gen)
            self._graph.pycode_gen.gen_call_function(
                argc=if_fn.__code__.co_argcount
            )
            self._graph.pycode_gen.gen_return()
        else:
            insert_index = len(self._graph.pycode_gen._instructions) - 1
            self._graph.pycode_gen.gen_return()

        if else_fn is not None:
            self._graph.pycode_gen.gen_load_object(
                else_fn, else_fn.__code__.co_name
            )
            jump_to = self._graph.pycode_gen._instructions[-1]
            for name in else_inputs:
                self.get_var(name).reconstruct(self._graph.pycode_gen)
            self._graph.pycode_gen.gen_call_function(
                argc=else_fn.__code__.co_argcount
            )
            self._graph.pycode_gen.gen_return()
        else:
            self._graph.pycode_gen.gen_return()
            jump_to = self._graph.pycode_gen._instructions[-1]

        self._graph.pycode_gen._insert_instr(
            insert_index, instr.opname, jump_to=jump_to
        )

        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn

    def transform(self):
        self.run()
        if self.new_code is None:
            raise InnerError("OpExecutor return a empty new_code.")
        return self.new_code, self.guard_fn

    def _fallback_in_for_loop(self, iterator, for_iter):
        '''
        for_iter: the FOR_ITER opcode

        need find out opcodes which unpack value from FOR_ITER, by analysing stack

        case 1:
            for i in iter:

            FOR_ITER
            STORE_FAST i

        case 2:
            for i,j in iter:

            FOR_ITER
            UNPACK_SEQUENCE 2
            STORE_FAST i
            STORE_FAST j

        TODO: check var is in globals or builtins, only locals considered now
        '''
        loop_body_start_idx = self.indexof(for_iter) + 1
        curent_stack = 1

        while True:
            if loop_body_start_idx >= len(self._instructions):
                raise InnerError("Can not balance stack in loop body.")
            cur_instr = self._instructions[loop_body_start_idx]
            # do not consider jump instr
            stack_effect = dis.stack_effect(
                cur_instr.opcode, cur_instr.arg, jump=False
            )
            curent_stack += stack_effect
            loop_body_start_idx += 1
            if curent_stack == 0:
                break

        pycode_gen = PyCodeGen(self._frame)
        loop_body, loop_inputs = pycode_gen.gen_loop_body_between(
            for_iter, loop_body_start_idx, self.indexof(for_iter.jump_to)
        )

        after_loop_fn, fn_inputs = self._create_resume_fn(
            self.indexof(for_iter.jump_to)
        )

        # 1. part before for-loop, start compile
        ret_names = [name for name in loop_inputs if name in self._locals]
        ret_vars = [self._locals[name] for name in ret_names]
        self._graph.start_compile(*ret_vars)
        for _ in ret_vars:
            self._graph.pycode_gen.pop_instr()

        # 2. restore vars
        for idx in range(len(ret_names)):
            ret_vars[idx].reconstruct(self._graph.pycode_gen)
            self._graph.pycode_gen.gen_store_fast(ret_names[idx])

        # 3. load iterator to stack
        iterator.reconstruct(self._graph.pycode_gen)

        # 4. gen FOR_ITER and unpack data
        self._graph.pycode_gen.extend_instrs(
            self._instructions[self.indexof(for_iter) : loop_body_start_idx]
        )

        # 5. call loop body
        # 5.1 load loop body
        self._graph.pycode_gen.gen_load_object(
            loop_body, loop_body.__code__.co_name
        )

        # 5.2 load loop body inputs
        def update_locals(name, variable):
            self._locals[name] = variable
            return variable

        for name in loop_inputs[:-1]:
            self._graph.pycode_gen.gen_load_fast(name)

        # 5.3 load break flag
        self._graph.pycode_gen.gen_load_const(True)

        # 5.4 call loop body
        self._graph.pycode_gen.gen_call_function(
            argc=loop_body.__code__.co_argcount
        )

        # 5.5 unpack and store retval, keep break_flag in stack
        self._graph.pycode_gen.gen_unpack_sequence(len(loop_inputs))

        for name in loop_inputs[:-1]:
            self._graph.pycode_gen.gen_store_fast(name)

        # 6. add jump if break
        jump_if_break = self._graph.pycode_gen._add_instr("POP_JUMP_IF_FALSE")

        # 7. add JUMP_ABSOLUTE to FOR_ITER
        self._graph.pycode_gen._add_instr("JUMP_ABSOLUTE", jump_to=for_iter)
        nop = self._graph.pycode_gen._add_instr("NOP")
        for_iter.jump_to = nop
        jump_if_break.jump_to = nop

        # 8. call after_loop_fn
        self._graph.pycode_gen.gen_load_object(
            after_loop_fn, after_loop_fn.__code__.co_name
        )

        for name in fn_inputs:
            self._graph.pycode_gen.gen_load_fast(name)

        self._graph.pycode_gen.gen_call_function(
            argc=after_loop_fn.__code__.co_argcount
        )

        self._graph.pycode_gen.gen_return()
        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn

    def _inline_call_for_loop(self, iterator, for_iter):
        # TODO: update globals builtins
        pycode_gen = PyCodeGen(self._frame)
        fn, inputs = pycode_gen.gen_for_loop_fn_between(
            iterator, self.indexof(for_iter), self.indexof(for_iter.jump_to)
        )
        fn = UserDefinedFunctionVariable(fn, self._graph, DummyTracker([]))
        input_vars = [self._locals[name] for name in inputs[:-1]] + [iterator]
        ret = fn(*input_vars)
        for name, val in zip(inputs[:-1], ret[:-1]):
            self._locals[name] = val

    def FOR_ITER(self, instr):
        iterator = self.pop()
        assert isinstance(iterator, IterVariable)
        backup_iter_idx = None

        start = self.indexof(instr)
        end = self.indexof(instr.jump_to)
        for i in range(start, end):
            if self._instructions[i].opname == "RETURN_VALUE":
                return Stop()

        # TODO need support TensorIterVariable.next
        try:
            if not isinstance(
                iterator, (SequenceIterVariable, DictIterVariable)
            ):
                raise BreakGraphError()
            backup_iter_idx = iterator.idx
            self._inline_call_for_loop(iterator, instr)
            self._lasti = self.indexof(instr.jump_to)
        except BreakGraphError:
            if backup_iter_idx:
                iterator.idx = backup_iter_idx
            self._fallback_in_for_loop(iterator, instr)
            return Stop()
