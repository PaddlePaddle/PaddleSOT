from __future__ import annotations

import collections
import dis
import functools
import inspect
import operator
import traceback
import types
from typing import Callable, List, Optional, Tuple

from ...utils import (
    BreakGraphError,
    InnerError,
    NotImplementException,
    Singleton,
    is_strict_mode,
    log,
    log_do,
)
from ..instruction_utils import Instruction, analysis_inputs, get_instructions
from .function_graph import FunctionGraph
from .guard import Guard
from .instr_flag import FORMAT_VALUE_FLAG as FV
from .instr_flag import MAKE_FUNCTION_FLAG as MF
from .pycode_generator import PyCodeGen
from .tracker import (
    BuiltinTracker,
    CellTracker,
    ConstTracker,
    DanglingTracker,
    DummyTracker,
    GetItemTracker,
    GetIterTracker,
    GlobalTracker,
    LocalTracker,
)
from .variable_dispatch import (
    operator_BAD,
    operator_exception_match,
    operator_in,
    operator_not_in,
)
from .variables import (
    BuiltinVariable,
    CellVariable,
    ConstantVariable,
    ContainerVariable,
    DictIterVariable,
    DictVariable,
    DummyVariable,
    IterVariable,
    ListVariable,
    MethodVariable,
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
    "==": operator.eq,
    "!=": operator.ne,
    "is not": operator.is_not,
    "is": operator.is_,
    "in": operator_in,
    "not in": operator_not_in,
    "exception match": operator_exception_match,
    "BAD": operator_BAD,
}


class Stop:
    pass


@Singleton
class InstructionTranslatorCache:
    """
    A singleton class that implements a cache for translated instructions.
    This cache is used to store previously translated instructions along with their corresponding guard functions.

    Attributes:
        cache (dict): A dictionary that maps code objects to tuples of a cache getter function and a list of guarded functions.
        translate_count (int): The count of how many instructions have been translated. It is used to test whether the cache hits.
    """

    cache: dict[types.CodeType, tuple[CacheGetter, GuardedFunctions]]
    translate_count: int

    def __init__(self):
        self.cache = {}
        self.translate_count = 0

    def clear(self):
        """
        Clears the cache and resets the translate count.
        """
        self.cache.clear()
        self.translate_count = 0

    def __call__(self, frame: types.FrameType, **kwargs) -> CustomCode | None:
        code: types.CodeType = frame.f_code
        if code not in self.cache:
            cache_getter, (new_code, guard_fn) = self.translate(frame, **kwargs)
            self.cache[code] = (cache_getter, [(new_code, guard_fn)])
            if cache_getter == self.skip:
                return None
            return CustomCode(new_code, False)
        cache_getter, guarded_fns = self.cache[code]
        return cache_getter(frame, guarded_fns)

    def lookup(self, **kwargs):
        def impl(
            frame: types.FrameType, guarded_fns: GuardedFunctions
        ) -> CustomCode | None:
            """
            Looks up the cache for a matching code object and returns a custom code object if a matching guard function is found, otherwise None.

            Args:
                frame (types.FrameType): The frame whose code object needs to be looked up in the cache.
                guarded_fns (GuardedFunctions): The list of guarded functions associated with the code object.

            Returns:
                CustomCode | None: The custom code object if a matching guard function is found, otherwise None.
            """
            for code, guard_fn in guarded_fns:
                try:
                    if guard_fn(frame):
                        log(
                            3,
                            f"[Cache]: Cache hit, Guard is {guard_fn.expr if hasattr(guard_fn, 'expr') else 'None'}\n",
                        )
                        return CustomCode(code, False)
                except Exception as e:
                    log(3, f"[Cache]: Guard function error: {e}\n")
                    continue
            cache_getter, (new_code, guard_fn) = self.translate(frame, **kwargs)
            guarded_fns.append((new_code, guard_fn))
            return CustomCode(new_code, False)

        return impl

    def skip(
        self, frame: types.FrameType, guarded_fns: GuardedFunctions
    ) -> CustomCode | None:
        """
        Skips the frame.

        Args:
            frame (types.FrameType): The frame to be skipped.
            guarded_fns (GuardedFunctions): The list of guarded functions associated with the skipped frame.

        Returns:
            CustomCode | None: None.
        """
        log(3, f"[Cache]: Skip frame {frame.f_code.co_name}\n")
        return None

    def translate(
        self, frame: types.FrameType, **kwargs
    ) -> tuple[CacheGetter, GuardedFunction]:
        """
        Translates the given frame's code object and returns the cache getter function and a guarded function for the translated code object.

        Args:
            frame (types.FrameType): The frame whose code object needs to be translated.

        Returns:
            tuple[CacheGetter, GuardedFunction]: The cache getter function and a guarded function for the translated code object.
        """
        code: types.CodeType = frame.f_code
        log(3, "[Cache]: Cache miss\n")
        self.translate_count += 1

        result = start_translate(frame, **kwargs)
        if result is None:
            return self.skip, (code, dummy_guard)

        new_code, guard_fn = result
        return self.lookup(**kwargs), (new_code, guard_fn)


def start_translate(frame: types.FrameType, **kwargs) -> GuardedFunction | None:
    """
    Starts the translation process for the given frame and returns the translated code object and its guard function, or None if translation fails.

    Args:
        frame: The frame to be translated.

    Returns:
        GuardedFunction | None: The translated code object and its guard function, or None if translation fails.
    """
    simulator = OpcodeExecutor(frame, **kwargs)
    try:
        log(3, f"OriginCode: {simulator._code}\n")
        log_do(3, lambda: dis.dis(simulator._code))
        new_code, guard_fn = simulator.transform()
        log(3, f"NewCode: {new_code}\n")
        log_do(3, lambda: dis.dis(new_code))
        return new_code, guard_fn
    # TODO(zrr1999): InnerError maybe place before (NotImplementException, BreakGraphError)
    # TODO(0x45f): handle BreakGraphError to trigger fallback
    except (NotImplementException, BreakGraphError) as e:
        if is_strict_mode():
            raise
        log(
            2,
            f"Unsupport Frame is {frame.f_code}, error message is: \n"
            + '\n'.join(traceback.format_exception_only(type(e), e)),
        )

        # NOTE: If resume fn need fallback, we should replace DummyVariable using NULL otherwise will fail to run
        py_codegen = PyCodeGen(frame)
        return py_codegen.replace_dummy_variable()
    except Exception as e:
        raise InnerError(OpcodeExecutorBase.error_message_summary(e)) from e


def tos_op_wrapper(fn: Callable):
    """
    A decorator function that wraps an opcode operation and applies certain functionality to it.

    Args:
        fn: The opcode operation to be wrapped.

    Returns:
        The wrapped opcode operation.
    """
    nargs = len(inspect.signature(fn).parameters)

    @call_break_graph_decorator(push_n=1)
    def inner(self: OpcodeExecutorBase, instr: Instruction):
        args = self.pop_n(nargs)
        res = BuiltinVariable(fn, graph=self._graph, tracker=DanglingTracker())(
            *args
        )
        self.push(res)

    return inner


def tos_inplace_op_wrapper(fn: Callable):
    """
    A decorator function that wraps an inplace opcode operation and applies certain functionality to it.

    Args:
        fn: The inplace opcode operation to be wrapped.

    Returns:
        The wrapped inplace opcode operation.

    """

    @call_break_graph_decorator(push_n=1)
    def inner(self: OpcodeExecutorBase, instr: Instruction):
        """
        Inner function that represents the wrapped inplace opcode operation.

        Args:
            self: The instance of the OpcodeExecutorBase class.
            instr: The instruction to be executed.

        """
        args = self.pop_n(2)
        res = BuiltinVariable(fn, graph=self._graph, tracker=DanglingTracker())(
            *args
        )
        res.debug_name = args[0].debug_name
        self.push(res)

    return inner


def jump_break_graph_decorator(normal_jump):
    """
    A decorator function that breaks off the graph when a JUMP-related instruction is encountered.

    Args:
        normal_jump: The normal jump operation.

    Returns:
        The wrapped jump operation.

    """

    def inner(self: OpcodeExecutor, instr: Instruction):
        result = self.peek()
        if isinstance(result, TensorVariable):
            self.pop()
            # fallback when in OpcodeExecutor
            # raise error in OpcodeInlineExecutor
            self._break_graph_in_jump(result, instr)
            return Stop()
        else:
            return normal_jump(self, instr)

    return inner


def call_break_graph_decorator(push_n: int):
    """
    A decorator function that breaks off the graph when a function CALL instruction is encountered.

    Args:
        push_n: The number of arguments to be pushed onto the stack.

    Returns:
        The decorated function.

    """

    def decorate(call_fn: Callable):
        @functools.wraps(call_fn)
        def wrapper(self: OpcodeExecutor, instr: Instruction):
            origin_stack = list(self._stack)
            try:
                return call_fn(self, instr)
            except BreakGraphError as e:
                if isinstance(self, OpcodeExecutor):
                    log(3, f"[BreakGraph] call function Break graph: {e}\n")
                    self._break_graph_in_call(origin_stack, instr, push_n)
                    return Stop()
                else:
                    raise e

        return wrapper

    return decorate


def fallback_when_occur_error(fn: Callable):
    """
    A decorator function that provides fallback behavior when an error occurs during graph processing.

    Args:
        fn: The function to be wrapped.

    Returns:
        The wrapped function.

    """

    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise NotImplementException(
                f'An exception occurred when processing break graph, fallback to dygraph, error message is: \n{type(e)} : {e}\n'
            )

    return inner


class OpcodeExecutorBase:
    """
    Base class for executing opcode instructions.

    The OpcodeExecutorBase class provides methods and functionality to execute opcode instructions.

    If you want to learn more about Python instructions, see https://docs.python.org/3/library/dis.html for details.

    Args:
        code: The bytecode of the function to be executed.
        graph: The function graph.

    Attributes:
        call_stack (list[OpcodeExecutorBase]): A list to keep track of the call stack.
        _stack (list[VariableBase]): The stack used for storing variables during execution.
        _co_consts: List to store constants.
        _locals (dict): Dictionary to store local variables.
        _globals (dict): Dictionary to store global variables.
        _builtins (dict): Dictionary to store built-in variables.
        _lasti (int): Index of the last executed instruction.
        _code (types.CodeType): The code object to be executed.
        _instructions: Iterator of opcode instructions.
        _graph (FunctionGraph): The function graph representing the code.
        _current_line: The current line number of the execution.
        new_code: Placeholder for new code (to be generated by PyCodeGen).
        guard_fn: Placeholder for guard function.
        _name (str): Name of the executor.

    """

    call_stack: list[OpcodeExecutorBase] = []

    def __init__(self, code: types.CodeType, graph: FunctionGraph):
        OpcodeExecutorBase.call_stack.append(self)
        # fake env for run, new env should be gened by PyCodeGen
        self._stack: list[VariableBase] = []
        self._co_consts = []
        self._locals = {}
        self._globals = {}
        self._builtins = {}
        self._cells = {}  # position to put cells
        self._lasti = 0  # idx of instruction list
        self._code = code
        self._instructions = get_instructions(self._code)
        self._graph = graph
        self._current_line: int = -1
        self.new_code: types.CodeType | None = None
        self.guard_fn = None
        self._name = "Executor"
        self._prepare_virtual_env()

    def print_sir(self):
        """
        Prints the Static Instruction Representation (SIR) in the executor.

        """
        print(self._graph.sir_ctx.TOS)

    def _prepare_virtual_env(self):
        """
        Prepares the virtual environment for the executor.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("Please implement virtual_env.")

    def _break_graph_in_jump(self, result, instr: Instruction):
        """
        Breaks the graph in JUMP instructions.

        Args:
            result: The execution result.
            instr: The jump instruction.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError()

    def transform(self):
        """
        Abstract method need to be implemented to symbolic translate each instruction.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError()

    def get_var(self, name: str):
        """
        Gets the variable with the given name.

        Args:
            name: The name of the variable.

        Returns:
            The variable.

        Raises:
            InnerError: If the variable cannot be found.

        """
        if name in self._locals.keys():
            return self._locals[name]
        elif name in self._globals.keys():
            return self._globals[name]
        elif name in self._builtins.keys():
            return self._builtins[name]
        elif name in self._cells.keys():  # in closure
            return self._cells[name].get_value()
        else:
            raise InnerError(f'Can not get var: {name}')

    def pop_call_stack_until_self(self):
        """
        Pops the call stack until the current executor.

        """
        assert (
            self in OpcodeExecutorBase.call_stack
        ), f"{self} not in call stack"
        while OpcodeExecutorBase.call_stack.pop() is not self:
            pass

    @staticmethod
    def error_message_summary(original_error: Exception) -> str:
        """
        Creates a summary of the error message during execution.

        Args:
            original_error: The original error.

        Returns:
            The summary error message.

        """
        indent = 2 * " "
        message_lines = ["In simulate execution:", ""]
        for current_simulator in OpcodeExecutorBase.call_stack:
            code = current_simulator._code
            current_line = current_simulator._current_line
            lines, start = inspect.getsourcelines(code)
            real_name = code.co_name
            message_lines.append(
                f"{indent}  File \"{code.co_filename}\", line {current_line}, in {real_name}"
            )
            if current_line != -1:
                message_lines.append(
                    f"{indent}  {lines[current_line-start].rstrip()}"
                )
        error_message = traceback.format_exception_only(
            type(original_error), original_error
        )
        for line in error_message:
            message_lines.append(f"{indent}  {line}")
        return "\n".join(message_lines)

    def run(self):
        """
        Executes the opcode.

        """
        log(3, f"start execute opcode: {self._code}\n")
        self._lasti = 0
        while True:
            if self._lasti >= len(self._instructions):
                raise InnerError("lasti out of range, InnerError.")
            cur_instr = self._instructions[self._lasti]
            self._lasti += 1
            is_stop = self.step(cur_instr)
            if is_stop:
                self.pop_call_stack_until_self()
                break

    def step(self, instr: Instruction):
        """
        Executes a single step of the opcode.

        Args:
            instr: The instruction to be executed.

        Returns:
            True if execution should stop, False otherwise.

        Raises:
            NotImplementException: If the opcode is not supported.

        """
        if instr.starts_line is not None:
            self._current_line = instr.starts_line
        if not hasattr(self, instr.opname):
            raise NotImplementException(
                f"opcode: {instr.opname} is not supported."
            )
        log_message = f"[Translate {self._name}]: (line {self._current_line:>3}) {instr.opname:<12} {instr.argval}, stack is {self._stack}\n"
        log(3, log_message)
        code_file = self._code.co_filename
        code_line = self._current_line
        from ..breakpoint import BreakpointManager

        if BreakpointManager().hit(code_file, code_line):
            BreakpointManager().locate(self)
            print(log_message)
            breakpoint()  # breakpoint for debug
        return getattr(self, instr.opname)(instr)  # run single step.

    def indexof(self, instr: Instruction):
        """
        Gets the index of the instruction.

        Args:
            instr: The instruction.

        Returns:
            The index of the instruction.

        """
        return self._instructions.index(instr)

    def pop(self) -> VariableBase:
        """
        Pops the top value from the stack.

        Returns:
            The popped value.

        """
        return self._stack.pop()

    def peek(self) -> VariableBase:
        """
        Peeks at the top value of the stack.

        Returns:
            The value at the top of the stack.

        """
        return self._stack[-1]

    def peek_n(self, n: int) -> list[VariableBase]:
        """
        Peeks at the top n values of the stack.

        Args:
            n: The number of values to peek.

        Returns:
            A list of the top n values of the stack.

        """
        return self._stack[-n:]

    def pop_n(self, n: int) -> list[VariableBase]:
        """
        Pops the top n values from the stack.

        Args:
            n: The number of values to pop.

        Returns:
            A list of the popped values.

        """
        if n == 0:
            return []
        retval = self._stack[-n:]
        self._stack[-n:] = []
        return retval

    def push(self, val: VariableBase):
        """
        Pushes a value onto the stack.

        Args:
            val: The value to be pushed.

        Raises:
            AssertionError: If the value is not an instance of VariableBase or is a dangling variable.

        """
        assert isinstance(
            val, (VariableBase)
        ), f"value: {val}, type shoule be VariableBase(or derived), but get {type(val)}"
        assert not isinstance(val.tracker, DanglingTracker) or isinstance(
            val, (DummyVariable, CellVariable)
        ), f"dangling variable {val} should not be pushed into stack."
        self._stack.append(val)

    def DUP_TOP(self, instr: Instruction):
        self.push(self.peek())

    def DUP_TOP_TWO(self, instr: Instruction):
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

    def POP_TOP(self, instr: Instruction):
        self.pop()

    def ROT_TWO(self, instr: Instruction):
        self._rot_top_n(2)

    def ROT_THREE(self, instr: Instruction):
        self._rot_top_n(3)

    def ROT_FOUR(self, instr: Instruction):
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

    def BINARY_SUBSCR(self, instr: Instruction):
        key = self.pop()
        container = self.pop()
        assert isinstance(key, VariableBase)
        self._graph.add_global_guarded_variable(key)
        self.push(
            BuiltinVariable(operator.getitem, self._graph, DanglingTracker())(
                container, key.get_value()
            )
        )

    # inplace operators
    # paddle variable do not have inplace operators. For example when call `y **= x`, will call var.__pow__
    INPLACE_POWER = tos_inplace_op_wrapper(operator.ipow)
    INPLACE_MULTIPLY = tos_inplace_op_wrapper(operator.imul)
    INPLACE_MATRIX_MULTIPLY = tos_inplace_op_wrapper(operator.imatmul)
    INPLACE_FLOOR_DIVIDE = tos_inplace_op_wrapper(operator.ifloordiv)
    INPLACE_TRUE_DIVIDE = tos_inplace_op_wrapper(operator.itruediv)
    INPLACE_MODULO = tos_inplace_op_wrapper(operator.imod)
    INPLACE_ADD = tos_inplace_op_wrapper(operator.iadd)
    INPLACE_SUBTRACT = tos_inplace_op_wrapper(operator.isub)
    INPLACE_LSHIFT = tos_inplace_op_wrapper(operator.ilshift)
    INPLACE_RSHIFT = tos_inplace_op_wrapper(operator.irshift)
    INPLACE_AND = tos_inplace_op_wrapper(operator.iand)
    INPLACE_OR = tos_inplace_op_wrapper(operator.ior)
    INPLACE_XOR = tos_inplace_op_wrapper(operator.ixor)

    def NOP(self, instr: Instruction):
        pass

    def LOAD_ATTR(self, instr: Instruction):
        attr_name = self._code.co_name[instr.arg]
        obj = self.pop()
        self.push(
            BuiltinVariable(
                getattr, graph=self._graph, tracker=DanglingTracker()
            )(obj, attr_name)
        )

    def LOAD_CONST(self, instr: Instruction):
        var = self._co_consts[instr.arg]
        self.push(var)

    def LOAD_CLOSURE(self, instr):
        namemap = self._code.co_cellvars + self._code.co_freevars
        name = namemap[instr.arg]
        self.push(self._cells[name])

    def LOAD_DEREF(self, instr):
        namemap = self._code.co_cellvars + self._code.co_freevars
        name = namemap[instr.arg]
        self.push(self._cells[name].get_value())

    def LOAD_FAST(self, instr: Instruction):
        varname = self._code.co_varnames[instr.arg]
        var = self._locals[varname]
        self.push(var)

    def LOAD_GLOBAL(self, instr: Instruction):
        name = self._code.co_names[instr.arg]
        if name in self._globals.keys():
            value = self._globals[name]
        else:
            value = self._builtins[name]
        self.push(value)

    def LOAD_METHOD(self, instr: Instruction):
        method_name = self._code.co_names[instr.arg]
        obj = self.pop()

        method = BuiltinVariable(
            getattr, graph=self._graph, tracker=DanglingTracker()
        )(obj, method_name)

        if isinstance(method, MethodVariable):
            # bound method, push the unbound method and the self
            self.push(method.fn)
            self.push(obj)
        else:
            # unbound method, push the dummy and the function
            self.push(DummyVariable())
            self.push(method)

    def STORE_DEREF(self, instr):
        namemap = self._code.co_cellvars + self._code.co_freevars
        name = namemap[instr.arg]
        self._cells[name].set_value(self.pop())

    def STORE_FAST(self, instr: Instruction):
        """
        TODO: side effect may happen
        """
        var = self.pop()
        name = self._code.co_varnames[instr.arg]
        var.debug_name = name
        self._locals[name] = var

    def STORE_GLOBAL(self, instr: Instruction):
        var = self.pop()
        name = self._code.co_names[instr.arg]
        var.debug_name = name
        self._locals[name] = var

    def STORE_SUBSCR(self, instr: Instruction):
        key = self.pop()
        container = self.pop()
        value = self.pop()
        assert isinstance(key, VariableBase)
        self._graph.add_global_guarded_variable(key)
        container[key.get_value()] = value
        value.debug_name = f"{container.debug_name}[{key.debug_name}]"

    def DELETE_SUBSCR(self, instr: Instruction):
        key = self.pop()
        container = self.pop()
        assert isinstance(key, VariableBase)
        self._graph.add_global_guarded_variable(key)
        BuiltinVariable(operator.delitem, self._graph, DanglingTracker())(
            container, key
        )

    def BUILD_LIST(self, instr: Instruction):
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

    def BUILD_TUPLE(self, instr: Instruction):
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

    def BUILD_STRING(self, instr: Instruction):
        count = instr.arg
        assert count <= len(
            self._stack
        ), f"OpExecutor want BUILD_STRING with size {count}, but current stack do not have enough elems."
        str_list = self.pop_n(count)
        new_str = ''
        for s in str_list:
            assert isinstance(s.value, str)
            new_str += s.value
        self.push(ConstantVariable.wrap_literal(new_str, self._graph))

    def BUILD_SLICE(self, instr: Instruction):
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

    def BUILD_MAP(self, instr: Instruction):
        map_size = instr.arg
        assert map_size * 2 <= len(
            self._stack
        ), f"OpExecutor want BUILD_MAP with size {map_size} * 2, but current stack do not have enough elems."
        val_for_dict = self.pop_n(map_size * 2)
        keys = val_for_dict[::2]
        values = val_for_dict[1::2]
        self.push(self.build_map(keys, values))

    def BUILD_CONST_KEY_MAP(self, instr: Instruction):
        map_size = instr.arg
        assert map_size + 1 <= len(
            self._stack
        ), f"OpExecutor want BUILD_CONST_KEY_MAP with size {map_size} + 1, but current stack do not have enough elems."
        keys = self.pop().get_items()
        assert len(keys) == map_size
        values = self.pop_n(map_size)
        self.push(self.build_map(keys, values))

    def build_seq_unpack(self, instr: Instruction):
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

    def BUILD_TUPLE_UNPACK_WITH_CALL(self, instr: Instruction):
        self.build_seq_unpack(instr)

    def BUILD_TUPLE_UNPACK(self, instr: Instruction):
        self.build_seq_unpack(instr)

    def BUILD_LIST_UNPACK(self, instr: Instruction):
        self.build_seq_unpack(instr)

    def BUILD_MAP_UNPACK(self, instr: Instruction):
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

    def BUILD_MAP_UNPACK_WITH_CALL(self, instr: Instruction):
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

    def CALL_FUNCTION(self, instr: Instruction):
        n_args = instr.arg
        assert n_args <= len(self._stack)
        args = self.pop_n(n_args)
        kwargs = {}
        fn = self.pop()
        ret = fn(*args, **kwargs)
        self.push(ret)

    def CALL_FUNCTION_KW(self, instr: Instruction):
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
        ret = fn(*args, **kwargs)
        self.push(ret)

    def CALL_FUNCTION_EX(self, instr: Instruction):
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
        ret = fn(*args, **kwargs)
        self.push(ret)

    def CALL_METHOD(self, instr: Instruction):
        n_args = instr.arg
        assert n_args <= len(self._stack)
        args = self.pop_n(n_args)
        self_var = self.pop()
        method = self.pop()
        if isinstance(method, DummyVariable):
            method = self_var
        else:
            args = [self_var] + args
        self.push(method(*args))

    def COMPARE_OP(self, instr: Instruction):
        op = instr.arg
        right, left = self.pop(), self.pop()
        self.push(
            BuiltinVariable(
                SUPPORT_COMPARE_OP[op], self._graph, DanglingTracker()
            )(left, right)
        )
        return

    def IS_OP(self, instr: Instruction):
        # It will only be 0 or 1
        assert instr.arg == 0 or instr.arg == 1
        right, left = self.pop(), self.pop()
        op = "is" if instr.arg == 0 else "is not"
        self.push(
            BuiltinVariable(
                SUPPORT_COMPARE_OP[op], self._graph, DanglingTracker()
            )(left, right)
        )

    def MAKE_FUNCTION(self, instr: Instruction):
        fn_name = self.pop()
        codeobj = self.pop()
        global_dict = self._globals

        related_list = [fn_name, codeobj]

        flag = instr.arg
        if flag & MF.MF_HAS_CLOSURE:
            # closure should be a tuple of Variables
            closure_variable = self.pop()
            assert isinstance(closure_variable, TupleVariable)
            closure = []
            for item in closure_variable.get_wrapped_items():
                closure.append(types.CellType())
                closure[-1].cell_contents = item
            closure = tuple(closure)
        else:
            closure = ()

        if flag & MF.MF_HAS_ANNOTATION:
            # can not set annotation in python env, skip it
            related_list.append(self.pop())

        if flag & MF.MF_HAS_KWDEFAULTS:
            raise NotImplementException(
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

    def GET_ITER(self, instr: Instruction):
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
            # TODO: source obj ? why not source_obj.__iter__()
            self.push(
                UserDefinedIterVariable(
                    source_obj, self._graph, GetIterTracker(source_obj)
                )
            )

    def JUMP_FORWARD(self, instr):
        self._lasti = self.indexof(instr.jump_to)

    def JUMP_ABSOLUTE(self, instr: Instruction):
        self._lasti = self.indexof(instr.jump_to)

    def CONTAINS_OP(self, instr: Instruction):
        # It will only be 0 or 1
        assert instr.arg == 0 or instr.arg == 1
        right, left = self.pop(), self.pop()
        op = "in" if instr.arg == 0 else "not in"
        self.push(
            BuiltinVariable(
                SUPPORT_COMPARE_OP[op], self._graph, DanglingTracker()
            )(left, right)
        )

    @jump_break_graph_decorator
    def JUMP_IF_FALSE_OR_POP(self, instr: Instruction):
        pred_obj = self.peek()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = not bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            else:
                self.pop()
            return
        raise NotImplementException(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @jump_break_graph_decorator
    def JUMP_IF_TRUE_OR_POP(self, instr: Instruction):
        pred_obj = self.peek()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            else:
                self.pop()
            return
        raise NotImplementException(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @jump_break_graph_decorator
    def POP_JUMP_IF_FALSE(self, instr: Instruction):
        pred_obj = self.pop()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = not bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            return
        raise NotImplementException(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @jump_break_graph_decorator
    def POP_JUMP_IF_TRUE(self, instr: Instruction):
        pred_obj = self.pop()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            return
        raise NotImplementException(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    def UNPACK_SEQUENCE(self, instr: Instruction):
        sequence = self.pop()

        '''
            TODO: To unpack iterator
            To unpack is easy, just like:
                seq = tuple(sequence.value)

            But what is the `source` when iterator returned a value ?
        '''
        if isinstance(sequence, TensorVariable):
            # TODO: If need to unpack a Tensor, should have different logic.
            raise NotImplementException("Unpack a iterator is not implemented.")
        elif isinstance(sequence, (ListVariable, TupleVariable)):
            seq = sequence.value
        else:
            raise NotImplementException(
                f"Unpack {sequence} is not implemented."
            )

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

    def FORMAT_VALUE(self, instr: Instruction):
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
            raise NotImplementException(
                f"Do not support format {type(value)} now"
            )

    # NOTE: This operation will generate SideEffects, and the mechanism has not been completed yet
    def DICT_UPDATE(self, instr: Instruction):
        dict_value = self.pop()
        assert instr.arg > 0
        BuiltinVariable(dict.update, self._graph, tracker=DanglingTracker())(
            self._stack[-instr.arg], dict_value
        )

    def DICT_MERGE(self, instr: Instruction):
        dict_value = self.pop()
        assert instr.arg > 0
        for key in dict_value.get_wrapped_items().keys():
            result = self._stack[-instr.arg].get_wrapped_items().get(key, None)
            if result is not None:
                raise InnerError(
                    f"got multiple values for keyword argument '{key}'"
                )
        BuiltinVariable(dict.update, self._graph, tracker=DanglingTracker())(
            self._stack[-instr.arg], dict_value
        )

    def LIST_APPEND(self, instr: Instruction):
        list_value = self.pop()
        assert instr.arg > 0
        BuiltinVariable(list.append, self._graph, tracker=DanglingTracker())(
            self._stack[-instr.arg], list_value
        )

    def LIST_EXTEND(self, instr: Instruction):
        list_value = self.pop()
        assert instr.arg > 0
        BuiltinVariable(list.extend, self._graph, tracker=DanglingTracker())(
            self._stack[-instr.arg], list_value
        )

    def LIST_TO_TUPLE(self, instr: Instruction):
        list_value = self.pop()
        self.push(
            TupleVariable(
                list_value.get_wrapped_items(),
                self._graph,
                DummyTracker([list_value]),
            )
        )


class OpcodeExecutor(OpcodeExecutorBase):
    """
    A class that represents an executor for opcode operations.

    Args:
        frame: The frame object.

    """

    def __init__(self, frame: types.FrameType, **kwargs):
        graph = FunctionGraph(frame, **kwargs)
        self._frame = frame
        self._name = "Executor"
        self.call_stack[:] = []
        super().__init__(frame.f_code, graph)

    def _prepare_virtual_env(self):
        """
        Prepare the virtual environment for execution by adding variables from locals, globals, builtins, and constants.

        """
        log(3, f"[Executor] code options: {self._frame.f_code.co_cellvars}\n")
        free_or_cell_vars = (
            self._frame.f_code.co_cellvars + self._frame.f_code.co_freevars
        )
        for name, value in self._frame.f_locals.items():
            tracker = (
                CellTracker(name)
                if name in free_or_cell_vars
                else LocalTracker(name)
            )
            self._locals[name] = VariableFactory.from_value(
                value, self._graph, tracker, debug_name=name
            )

        for name in free_or_cell_vars:
            # create a cell for each variable.
            self._cells[name] = CellVariable()  # put in cells.
            if name in self._locals:
                self._cells[name].set_value(self._locals[name])

        for name, value in self._frame.f_globals.items():
            self._globals[name] = VariableFactory.from_value(
                value, self._graph, GlobalTracker(name), debug_name=name
            )

        for name, value in self._frame.f_builtins.items():
            self._builtins[name] = VariableFactory.from_value(
                value, self._graph, BuiltinTracker(name), debug_name=name
            )

        for value in self._code.co_consts:
            self._co_consts.append(
                VariableFactory.from_value(
                    value, self._graph, ConstTracker(value)
                )
            )

    def _create_resume_fn(self, index, stack_size=0):
        """
        Create a resume function and its inputs at the specified index.

        Args:
            index: The index at which the resume function is created.
            stack_size: The size of the stack.

        Returns:
            The resume function and its inputs.

        """
        pycode_gen = PyCodeGen(self._frame)
        fn, inputs = pycode_gen.gen_resume_fn_at(index, stack_size)
        return fn, inputs

    @fallback_when_occur_error
    def _break_graph_in_jump(self, result: VariableBase, instr: Instruction):
        """
        Break the graph at a JUMP instruction.

        Args:
            result: The result variable of the jump instruction.
            instr: The jump instruction.

        """
        self._graph.add_global_guarded_variable(result)
        stack_size = len(self._stack)
        if_fn, if_inputs = self._create_resume_fn(
            self.indexof(instr) + 1, stack_size
        )
        else_fn, else_inputs = self._create_resume_fn(
            self.indexof(instr.jump_to), stack_size
        )

        # gen call static fn opcode
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
        # only pop the input of if/else resume fn, and keep the bool tensor result on the stack
        for _ in inputs_var:
            self._graph.pycode_gen.gen_pop_top()

        # gen call if/else resume fn opcode
        if if_fn is not None:
            self._graph.pycode_gen.gen_load_object(
                if_fn, if_fn.__code__.co_name
            )
            insert_index = len(self._graph.pycode_gen._instructions) - 1
            for stack_arg in self._stack:
                stack_arg.reconstruct(self._graph.pycode_gen)
            for name in if_inputs:
                self.get_var(name).reconstruct(self._graph.pycode_gen)
            self._graph.pycode_gen.gen_call_function(
                argc=if_fn.__code__.co_argcount,
                with_eval_frame=True,
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
            for stack_arg in self._stack:
                stack_arg.reconstruct(self._graph.pycode_gen)
            for name in else_inputs:
                self.get_var(name).reconstruct(self._graph.pycode_gen)
            self._graph.pycode_gen.gen_call_function(
                argc=else_fn.__code__.co_argcount,
                with_eval_frame=True,
            )
            self._graph.pycode_gen.gen_return()
        else:
            self._graph.pycode_gen.gen_return()
            jump_to = self._graph.pycode_gen._instructions[-1]

        # gen jump opcode
        self._graph.pycode_gen._insert_instr(
            insert_index, instr.opname, jump_to=jump_to
        )

        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn

    @fallback_when_occur_error
    def _break_graph_in_call(
        self, origin_stack: list[VariableBase], instr: Instruction, push_n: int
    ):
        """
        Break the graph at a CALL instruction.

        Args:
            origin_stack: The original stack.
            instr: The call instruction.
            push_n: The number of elements to be pushed onto the stack.

        """
        index = self.indexof(instr)
        self._stack = origin_stack

        # gen call static fn opcode
        ret_vars = [
            arg
            for arg in self._stack
            if isinstance(arg, (TensorVariable, ContainerVariable))
        ]
        resume_input_name = analysis_inputs(self._instructions, index + 1)
        ret_vars = ret_vars + [
            self.get_var(name)
            for name in resume_input_name
            if self.get_var(name) not in ret_vars
        ]
        self._graph.start_compile(*ret_vars)
        for _ in ret_vars:
            self._graph.pycode_gen.gen_pop_top()

        # gen graph break call fn opcode
        stack_effect = dis.stack_effect(instr.opcode, instr.arg)
        pop_n = push_n - stack_effect
        for i, stack_arg in enumerate(self._stack):
            # Avoid passing NULL as a parameter to the resume function
            if (
                isinstance(stack_arg, DummyVariable)
                and i < len(self._stack) - pop_n
            ):
                self._graph.pycode_gen.gen_load_object(
                    DummyVariable(), f'dummy_var{i}'
                )
            else:
                stack_arg.reconstruct(self._graph.pycode_gen)
        self._graph.pycode_gen.add_pure_instructions([instr])

        # gen call resume fn opcode
        self.pop_n(pop_n)
        stack_size = len(self._stack) + push_n
        resume_fn, _ = self._create_resume_fn(index + 1, stack_size)
        if resume_fn:
            self._graph.pycode_gen.gen_load_object(
                resume_fn, resume_fn.__code__.co_name
            )
            self._graph.pycode_gen.gen_rot_n(stack_size + 1)
            for name in resume_input_name:
                self._locals[name].reconstruct(self._graph.pycode_gen)
            self._graph.pycode_gen.gen_call_function(
                argc=resume_fn.__code__.co_argcount,
                with_eval_frame=True,
            )

        # gen RETURN_VALUE
        self._graph.pycode_gen.gen_return()

        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn

    def transform(self):
        self.run()
        if self.new_code is None:
            raise InnerError("OpExecutor return a empty new_code.")
        return self.new_code, self.guard_fn

    @fallback_when_occur_error
    def _break_graph_in_for_loop(
        self, iterator: VariableBase, for_iter: Instruction
    ):
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
            self.indexof(for_iter.jump_to), len(self._stack)
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
            argc=loop_body.__code__.co_argcount, with_eval_frame=True
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

        for stack_arg in self._stack:
            stack_arg.reconstruct(self._graph.pycode_gen)
        for name in fn_inputs:
            self._graph.pycode_gen.gen_load_fast(name)

        self._graph.pycode_gen.gen_call_function(
            argc=after_loop_fn.__code__.co_argcount, with_eval_frame=True
        )

        self._graph.pycode_gen.gen_return()
        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn

    def _inline_call_for_loop(
        self, iterator: VariableBase, for_iter: Instruction
    ):
        # TODO: update globals builtins
        pycode_gen = PyCodeGen(self._frame)
        fn, inputs = pycode_gen.gen_for_loop_fn_between(
            iterator, self.indexof(for_iter), self.indexof(for_iter.jump_to)
        )
        fn = UserDefinedFunctionVariable(
            fn,
            self._graph,
            DanglingTracker(),
        )
        input_vars = [self._locals[name] for name in inputs[:-1]] + [iterator]
        ret = fn(*input_vars)
        for name, val in zip(inputs[:-1], ret[:-1]):
            self._locals[name] = val

    def STORE_ATTR(self, instr):
        obj = self.pop()
        val = self.pop()
        key = self._code.co_names[instr.arg]
        if isinstance(obj, TensorVariable):
            # support tensor variable store attr, like:
            # t.stop_gradient = True
            obj.graph.call_tensor_method(
                "__setattr__",
                obj,
                VariableFactory().from_value(
                    key, self._graph, ConstTracker(key)
                ),
                val,
            )
        else:
            raise NotImplementException(
                f"SETATTR don't support {obj}.{key}={val}"
            )

    def FOR_ITER(self, instr):
        iterator = self.pop()
        backup_iter_idx = None

        start = self.indexof(instr)
        end = self.indexof(instr.jump_to)
        for i in range(start, end):
            if self._instructions[i].opname == "RETURN_VALUE":
                raise NotImplementException(
                    "Found RETURN_VALUE in for loop body."
                )

        self._graph.add_global_guarded_variable(iterator)
        # TODO need support TensorIterVariable.next
        try:
            if not isinstance(
                iterator, (SequenceIterVariable, DictIterVariable)
            ):
                raise BreakGraphError()
            backup_iter_idx = iterator.idx
            self._inline_call_for_loop(iterator, instr)
            self._lasti = self.indexof(instr.jump_to)
        except BreakGraphError as e:
            if backup_iter_idx:
                iterator.idx = backup_iter_idx
            self._break_graph_in_for_loop(iterator, instr)
            return Stop()

    @call_break_graph_decorator(push_n=1)
    def CALL_FUNCTION(self, instr: Instruction):
        super().CALL_FUNCTION(instr)

    @call_break_graph_decorator(push_n=1)
    def CALL_METHOD(self, instr: Instruction):
        super().CALL_METHOD(instr)

    @call_break_graph_decorator(push_n=1)
    def CALL_FUNCTION_KW(self, instr: Instruction):
        super().CALL_FUNCTION_KW(instr)

    @call_break_graph_decorator(push_n=1)
    def CALL_FUNCTION_EX(self, instr: Instruction):
        super().CALL_FUNCTION_EX(instr)

    @call_break_graph_decorator(push_n=1)
    def LOAD_ATTR(self, instr: Instruction):
        super().LOAD_ATTR(instr)

    @call_break_graph_decorator(push_n=1)
    def BINARY_SUBSCR(self, instr: Instruction):
        super().BINARY_SUBSCR(instr)

    def RETURN_VALUE(self, instr: Instruction):
        assert (
            len(self._stack) == 1
        ), f"Stack must have one element, but get {len(self._stack)} elements."
        ret_val = self.pop()
        self._graph.start_compile(ret_val)
        self._graph.pycode_gen.gen_return()
        self.new_code = self._graph.pycode_gen.gen_pycode()
        # self.guard_fn = lambda x: True
        self.guard_fn = self._graph.guard_fn
        return Stop()
