# This class is used for abstract code generation:
# We only need to care about what type of bytecode our code needs to generate,
# without worrying about the subscripts of bytecode instructions in the code option.

from __future__ import annotations

import dis
import sys
import types
from typing import TYPE_CHECKING

import opcode

from ...utils import (
    ResumeFnNameFactory,
    list_contain_by_id,
    list_find_index_by_id,
    no_eval_frame,
)
from ..instruction_utils import (
    analysis_inputs,
    gen_instr,
    get_instructions,
    instrs_info,
    modify_instrs,
    modify_vars,
)

if TYPE_CHECKING:
    from ..instruction_utils import Instruction


def get_pycode_attributes():
    """Code options for PyCodeObject"""
    # NOTE(SigureMo): The order should consistent with signature specified in code_doc
    # https://github.com/python/cpython/blob/3.8/Objects/codeobject.c#L416-L421
    pycode_attributes = [
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_filename",
        "co_name",
        "co_firstlineno",
    ]
    if sys.version_info >= (3, 10):
        pycode_attributes.append("co_linetable")
    else:
        pycode_attributes.append("co_lnotab")
    pycode_attributes += [
        "co_freevars",
        "co_cellvars",
    ]
    return pycode_attributes


PYCODE_ATTRIBUTES = get_pycode_attributes()


def gen_code_options(code):
    code_options = {}
    for k in PYCODE_ATTRIBUTES:
        val = getattr(code, k)
        if isinstance(val, tuple):
            val = list(val)
        code_options[k] = val
    return code_options


def gen_new_opcode(instrs: list[Instruction], code_options, keys):
    """Generate a new code object"""
    bytecode, linetable = assemble(instrs, code_options["co_firstlineno"])
    if sys.version_info >= (3, 10):
        # Python deprecated co_lnotab in 3.10, use co_linetable instead
        # https://peps.python.org/pep-0626/
        code_options["co_linetable"] = linetable
    else:
        code_options["co_lnotab"] = linetable
    code_options["co_code"] = bytecode
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    code_options["co_stacksize"] = stacksize(instrs)
    for key, val in code_options.items():
        if isinstance(val, list):
            code_options[key] = tuple(val)
    # code_options is a dict, use keys to makesure the input order
    return types.CodeType(*[code_options[k] for k in keys])


def assemble(
    instructions: list[Instruction], firstlineno: int
) -> tuple[bytes, bytes]:
    """list of instructions => bytecode & lnotab"""
    code = []
    linetable = []

    calc_linetable, update_cursor = create_linetable_calculator(firstlineno)

    for instr in instructions:
        # set lnotab
        if instr.starts_line is not None:
            linetable.extend(calc_linetable(instr.starts_line, len(code)))
            update_cursor(instr.starts_line, len(code))

        # get bytecode
        arg = instr.arg or 0
        code.extend((instr.opcode, arg & 0xFF))

    if sys.version_info >= (3, 10):
        # End hook for Python 3.10
        linetable.extend(calc_linetable(0, len(code)))

    return bytes(code), bytes(linetable)


def to_byte(num):
    """Convert negative number to unsigned byte"""
    if num < 0:
        num += 256
    return num


def create_linetable_calculator(firstlineno: int):
    cur_lineno = firstlineno
    cur_bytecode = 0
    line_offset = 0  # For Python 3.10

    def update_cursor(starts_line: int, code_length: int):
        nonlocal cur_lineno, cur_bytecode
        cur_bytecode = code_length
        cur_lineno = starts_line

    def calc_lnotab(starts_line: int, code_length: int):
        """Python 3.8, 3.9 lnotab calculation
        https://github.com/python/cpython/blob/3.9/Objects/lnotab_notes.txt
        """
        nonlocal cur_lineno, cur_bytecode
        line_offset = starts_line - cur_lineno
        byte_offset = code_length - cur_bytecode
        result = []

        while line_offset or byte_offset:
            line_offset_step = min(max(line_offset, -128), 127)
            byte_offset_step = min(max(byte_offset, 0), 255)
            result.extend((byte_offset_step, to_byte(line_offset_step)))
            line_offset -= line_offset_step
            byte_offset -= byte_offset_step
        return result

    def calc_linetable(starts_line: int, code_length: int):
        """Python 3.10 linetable calculation
        https://github.com/python/cpython/blob/3.10/Objects/lnotab_notes.txt
        """
        nonlocal cur_lineno, cur_bytecode, line_offset
        byte_offset = code_length - cur_bytecode
        result = []
        while line_offset or byte_offset:
            line_offset_step = min(max(line_offset, -127), 127)
            byte_offset_step = min(max(byte_offset, 0), 254)
            result.extend((byte_offset_step, to_byte(line_offset_step)))
            line_offset -= line_offset_step
            byte_offset -= byte_offset_step
        line_offset = starts_line - cur_lineno
        return result

    if sys.version_info >= (3, 10):
        return calc_linetable, update_cursor
    else:
        return calc_lnotab, update_cursor


def stacksize(instructions):
    # Two list below shows the possible stack size before opcode is called
    # The stack size might be different in different branch, so it has max and min
    max_stack = [float("-inf")] * len(instructions)
    min_stack = [float("inf")] * len(instructions)

    max_stack[0] = 0
    min_stack[0] = 0

    def update_stacksize(lasti, nexti, stack_effect):
        max_stack[nexti] = max(
            max_stack[nexti], max_stack[lasti] + stack_effect
        )
        min_stack[nexti] = min(
            min_stack[nexti], max_stack[lasti] + stack_effect
        )

    for idx in range(len(instructions)):
        instr = instructions[idx]

        if idx + 1 < len(instructions):
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=False)
            update_stacksize(idx, idx + 1, stack_effect)

        if instr.opcode in opcode.hasjabs or instr.opcode in opcode.hasjrel:
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=True)
            target_idx = instructions.index(instr.jump_to)
            update_stacksize(idx, target_idx, stack_effect)

    assert min(min_stack) >= 0
    return max(max_stack)


class PyCodeGen:
    """Helper to create new code object"""

    def __init__(self, frame):
        self._frame = frame
        self._origin_code = frame.f_code
        self._code_options = gen_code_options(self._origin_code)
        self._f_globals = frame.f_globals
        self._instructions = []
        self.objname_map = {}  # map from name to LOAD_GLOBAL index

    def gen_pycode(self):
        """
        return a new pycode, which is runnable.
        """
        modify_instrs(self._instructions)
        modify_vars(self._instructions, self._code_options)
        new_code = gen_new_opcode(
            self._instructions, self._code_options, PYCODE_ATTRIBUTES
        )
        return new_code

    def gen_resume_fn_at(self, index, stack_size=0):
        self._instructions = get_instructions(self._origin_code)
        # TODO(dev): could give an example code here?
        if self._instructions[index].opname == 'RETURN_VALUE':
            return None, set()
        inputs = analysis_inputs(self._instructions, index)
        fn_name = ResumeFnNameFactory().next()
        stack_arg_str = fn_name + '_stack_{}'
        self._instructions = (
            [
                gen_instr('LOAD_FAST', argval=stack_arg_str.format(i))
                for i in range(stack_size)
            ]
            + [gen_instr('JUMP_ABSOLUTE', jump_to=self._instructions[index])]
            + self._instructions
        )

        self._code_options['co_argcount'] = len(inputs) + stack_size
        # inputs should be at the front of the co_varnames
        self._code_options['co_varnames'] = tuple(
            [stack_arg_str.format(i) for i in range(stack_size)]
            + list(inputs)
            + [
                var_name
                for var_name in self._origin_code.co_varnames
                if var_name not in inputs
            ]
        )
        self._code_options['co_name'] = fn_name

        new_code = self.gen_pycode()
        fn = types.FunctionType(new_code, self._f_globals, fn_name)

        return fn, inputs

    def _gen_fn(self, inputs):
        # outputs is same as inputs, and they are always in locals
        for name in inputs:
            self.gen_load_fast(name)

        self.gen_build_tuple(len(inputs))
        self.gen_return()

        self._code_options['co_argcount'] = len(inputs)
        self._code_options['co_varnames'] = tuple(
            list(inputs)
            + [
                var_name
                for var_name in self._origin_code.co_varnames
                if var_name not in inputs
            ]
        )
        fn_name = ResumeFnNameFactory().next()
        self._code_options['co_name'] = fn_name
        new_code = self.gen_pycode()
        fn = types.FunctionType(new_code, self._f_globals, fn_name)
        return fn

    def gen_loop_body_between(self, for_iter, start, end):
        break_flag_name = "_break_flag"
        origin_instrs = get_instructions(self._origin_code)
        inputs = list(analysis_inputs(origin_instrs, start)) + [break_flag_name]

        # for balance the stack (the loop body will pop iter first before break or return)
        # this None is used for replace the iterator obj in stack top
        self.gen_load_const(None)

        # extend loop body main logic
        self.extend_instrs(origin_instrs[start:end])

        # break should jump to this nop
        nop_for_break = self._add_instr("NOP")

        # need do additional operates when break
        self.gen_load_const(False)
        self.gen_store_fast(break_flag_name)
        self.gen_load_const(None)  # keep stack balance

        # continue should jump to this nop
        nop_for_continue = self._add_instr("NOP")
        self.gen_pop_top()

        out_loop = for_iter.jump_to
        for instr in self._instructions:
            if instr.jump_to == for_iter:
                instr.jump_to = nop_for_continue
            if instr.jump_to == out_loop:
                instr.jump_to = nop_for_break

        return self._gen_fn(inputs), inputs

    def gen_for_loop_fn_between(self, iterator, start, end):
        origin_instrs = get_instructions(self._origin_code)
        inputs = list(analysis_inputs(origin_instrs, start)) + [iterator.id]
        self.gen_load_fast(iterator.id)
        self.extend_instrs(origin_instrs[start:end])
        for_iter = origin_instrs[start]
        out_loop_instr = origin_instrs[start].jump_to

        nop_for_continue = self._add_instr("NOP")
        jump = self._add_instr("JUMP_ABSOLUTE", jump_to=for_iter)
        nop_for_break = self._add_instr("NOP")

        for instr in self._instructions:
            if instr.jump_to == for_iter:
                instr.jump_to = nop_for_continue

            if instr.jump_to == out_loop_instr:
                instr.jump_to = nop_for_break

        jump.jump_to = for_iter

        return self._gen_fn(inputs), inputs

    def gen_load_const(self, value):
        # Python `list.index` will find an item equal to query, i.e. `query == item`
        # returns a value of True. Since `1 == True`, this will result in an incorrect
        # index. To avoid this problem, we use id for comparison.
        if not list_contain_by_id(self._code_options["co_consts"], value):
            self._code_options["co_consts"].append(value)
        idx = list_find_index_by_id(self._code_options["co_consts"], value)
        self._add_instr("LOAD_CONST", arg=idx, argval=value)

    def gen_load_global(self, name):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=name)

    def gen_load_object(self, obj, obj_name):
        if obj_name not in self.objname_map:
            self._f_globals[obj_name] = obj
            self._code_options["co_names"].append(obj_name)
            idx = len(self._code_options["co_names"]) - 1
            self.objname_map[obj_name] = idx
        idx = self.objname_map[obj_name]
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=obj_name)

    def gen_load_fast(self, name):
        if name not in self._code_options["co_varnames"]:
            self._code_options["co_varnames"].append(name)
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("LOAD_FAST", arg=idx, argval=name)

    def gen_load_attr(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_ATTR", arg=idx, argval=name)

    def gen_load_method(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_METHOD", arg=idx, argval=name)

    def gen_import_name(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("IMPORT_NAME", arg=idx, argval=name)

    def gen_push_null(self):
        # There is no PUSH_NULL bytecode before python3.11, so we push
        # a NULL element to the stack through the following bytecode
        self.gen_load_const(0)
        self.gen_load_const(None)
        self.gen_import_name('sys')
        self.gen_store_fast('sys')
        self.gen_load_fast('sys')
        self.gen_load_method('getsizeof')
        self._add_instr("POP_TOP")
        # TODO(dev): push NULL element to the stack through PUSH_NULL bytecode in python3.11

    def gen_store_fast(self, name):
        if name not in self._code_options["co_varnames"]:
            self._code_options["co_varnames"].append(name)
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("STORE_FAST", arg=idx, argval=name)

    def gen_subscribe(self):
        self._add_instr("BINARY_SUBSCR")

    def gen_build_tuple(self, count):
        self._add_instr("BUILD_TUPLE", arg=count, argval=count)

    def gen_build_list(self, count):
        self._add_instr("BUILD_LIST", arg=count, argval=count)

    def gen_build_map(self, count):
        self._add_instr("BUILD_MAP", arg=count, argval=count)

    def gen_unpack_sequence(self, count):
        self._add_instr("UNPACK_SEQUENCE", arg=count, argval=count)

    def gen_call_function(self, argc=0):
        self._add_instr("CALL_FUNCTION", arg=argc, argval=argc)

    def gen_pop_top(self):
        self._add_instr("POP_TOP")

    def gen_rot_n(self, n):
        if n <= 1:
            return
        if n <= 4:
            self._add_instr("ROT_" + ["TWO", "THREE", "FOUR"][n - 2])
        else:

            def rot_n_fn(n):
                vars = [f"var{i}" for i in range(n)]
                rotated = reversed(vars[-1:] + vars[:-1])
                fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
                fn = no_eval_frame(fn)
                fn.__name__ = f"rot_{n}_fn"
                return fn

            self.gen_build_tuple(n)
            self.gen_load_const(rot_n_fn(n))
            self.gen_rot_n(2)
            self._add_instr("CALL_FUNCTION_EX", arg=0)
            self.gen_unpack_sequence(n)

    def gen_return(self):
        self._add_instr("RETURN_VALUE")

    def add_pure_instructions(self, instructions):
        """
        add instructions and do nothing.
        """
        self._instructions.extend(instructions)

    def _add_instr(self, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.append(instr)
        return instr

    def _insert_instr(self, index, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.insert(index, instr)

    def pprint(self):
        print(instrs_info(self._instructions))

    def extend_instrs(self, instrs):
        self._instructions.extend(instrs)

    def pop_instr(self):
        self._instructions.pop()

    def replace_dummy_variable(self):
        from .variables.basic import DummyVariable

        instructions = get_instructions(self._origin_code)
        has_dummy_variable = False
        for instr in instructions:
            if (
                instr.opname == 'LOAD_FAST'
                and instr.argval in self._frame.f_locals.keys()
                and isinstance(
                    self._frame.f_locals[instr.argval], DummyVariable
                )
            ):
                has_dummy_variable = True
                self._frame.f_locals[instr.argval].reconstruct(self)
            else:
                self.add_pure_instructions([instr])

        if has_dummy_variable:
            new_code = self.gen_pycode()
            return new_code, lambda frame: True
        else:
            return None
