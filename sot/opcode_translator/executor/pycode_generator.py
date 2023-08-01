# This class is used for abstract code generation:
# We only need to care about what type of bytecode our code needs to generate,
# without worrying about the subscripts of bytecode instructions in the code option.

from __future__ import annotations

import dis
import os
import re
import sys
import types
from typing import TYPE_CHECKING

import opcode

import paddle

from ...utils import (
    InnerError,
    NotImplementException,
    OrderedSet,
    ResumeFnNameFactory,
    list_contain_by_id,
    list_find_index_by_id,
    no_eval_frame,
)
from ..instruction_utils import (
    analysis_inputs,
    analysis_inputs_outputs,
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
    if not code_options['co_name'].startswith("#"):
        code_options[
            'co_name'
        ] = f"#{code_options['co_name']}_{hex(hash(code) & 0xFFFFF)[2:]:0>5}"
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

    max_stack[0] = 0

    queue = []
    queue.append(0)

    def update_stacksize(lasti, nexti, stack_effect):
        old_max = max_stack[nexti]
        max_stack[nexti] = max(
            max_stack[nexti], max_stack[lasti] + stack_effect
        )
        if old_max != max_stack[nexti]:
            if nexti not in queue:  # may be slow, we can use a flag.
                queue.append(nexti)

    while len(queue) > 0:
        idx = queue[0]
        del queue[0]
        instr = instructions[idx]
        opname = instr.opname
        if idx + 1 < len(instructions) and instr.opname not in [
            'JUMP_ABSOLUTE',
            "JUMP_FORWARD",
            "JUMP_BACKWRAD",
        ]:
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=False)
            update_stacksize(idx, idx + 1, stack_effect)

        if instr.opcode in opcode.hasjabs or instr.opcode in opcode.hasjrel:
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=True)
            target_idx = instructions.index(instr.jump_to)
            update_stacksize(idx, target_idx, stack_effect)

    # assert min(min_stack) >= 0 # min_stack may be a negative number when try: except is got.
    return max(max_stack)


class PyCodeGen:
    """Helper to create new code object"""

    def __init__(self, frame, disable_eval_frame=False):
        self._frame = frame
        self._origin_code = frame.f_code
        self._code_options = gen_code_options(self._origin_code)
        self._f_globals = frame.f_globals
        self._instructions = []
        self.objname_map = {}  # map from name to LOAD_GLOBAL index
        self.disable_eval_frame = disable_eval_frame
        if self.disable_eval_frame:
            self.gen_disable_eval_frame()

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
            return None, OrderedSet()
        inputs = analysis_inputs(self._instructions, index)
        fn_name = ResumeFnNameFactory().next()
        stack_args_list = fn_name + '_list'
        header = []
        for i in range(stack_size):
            header.append(gen_instr('LOAD_FAST', argval=stack_args_list))
            header.append(gen_instr('LOAD_CONST', argval=0))
            header.append(gen_instr('BINARY_SUBSCR', argval=None))
            header.append(gen_instr('LOAD_FAST', argval=stack_args_list))
            header.append(gen_instr('LOAD_CONST', argval=0))
            header.append(gen_instr('DELETE_SUBSCR', argval=None))

        for name in list(inputs):
            header.append(gen_instr('LOAD_FAST', argval=stack_args_list))
            header.append(gen_instr('LOAD_CONST', argval=0))
            header.append(gen_instr('BINARY_SUBSCR', argval=None))
            header.append(gen_instr('STORE_FAST', argval=name))

        self._instructions = (
            header
            + [gen_instr('JUMP_ABSOLUTE', jump_to=self._instructions[index])]
            + self._instructions
        )
        if 0 not in self._code_options['co_consts']:
            self._code_options['co_consts'] += [0]
        self._code_options['co_argcount'] = 1
        # inputs should be at the front of the co_varnames
        self._code_options['co_varnames'] = list(
            [stack_args_list]
            + list(inputs)
            + [
                var_name
                for var_name in self._origin_code.co_varnames
                if var_name not in inputs
            ]
        )
        self._code_options[
            'co_name'
        ] = f"#{fn_name}@{self._code_options['co_name'][1:]}"

        new_code = self.gen_pycode()
        if len(new_code.co_freevars) > 0:
            raise NotImplementException(
                "Break graph in closure is not support."
            )
        fn = types.FunctionType(new_code, self._f_globals, fn_name)

        return fn, inputs

    def gen_disable_eval_frame(self):
        if os.environ.get('CLEAN_CODE', None) is not None:
            return
        self.gen_load_object(
            paddle.fluid.core.set_eval_frame, "paddle_set_eval_frame_fn"
        )
        self.gen_load_const(None)
        self.gen_call_function(1)
        self.gen_store_fast("___old_eval_frame")

    def gen_enable_eval_frame(self):
        if os.environ.get('CLEAN_CODE', None) is not None:
            return
        self.gen_load_object(
            paddle.fluid.core.set_eval_frame, "paddle_set_eval_frame_fn"
        )
        self.gen_load_fast("___old_eval_frame")
        self.gen_call_function(1)
        self.gen_pop_top()

    def create_fn_with_specific_io(self, inputs, outputs):
        '''
        generate the return value part of function, and return function object
        the main codes should be created before call create_fn_with_specific_io
        '''
        for name in outputs:
            self.gen_load(name)
        self.gen_build_tuple(len(outputs))
        self._code_options['co_argcount'] = len(inputs)
        self._code_options['co_varnames'] = list(
            list(inputs)
            + [
                var_name
                for var_name in self._origin_code.co_varnames
                if var_name not in inputs
            ]
        )
        self.gen_return()
        fn_name = ResumeFnNameFactory().next()
        self._code_options[
            'co_name'
        ] = f"#{fn_name}@{self._code_options['co_name'][1:]}"
        new_code = self.gen_pycode()
        if len(new_code.co_freevars) > 0:
            raise NotImplementException(
                "Break graph in closure is not support."
            )
        fn = types.FunctionType(new_code, self._f_globals, fn_name)
        return fn

    def gen_loop_body_between(self, for_iter, start, end):
        break_flag_name = "_break_flag"
        origin_instrs = get_instructions(self._origin_code)
        inputs = list(analysis_inputs_outputs(origin_instrs, start, end)) + [
            break_flag_name
        ]

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

        # outputs is the same as inputs
        return self.create_fn_with_specific_io(inputs, inputs), inputs

    def gen_load_const(self, value):
        # Python `list.index` will find an item equal to query, i.e. `query == item`
        # returns a value of True. Since `1 == True`, this will result in an incorrect
        # index. To avoid this problem, we use id for comparison.
        if not list_contain_by_id(self._code_options["co_consts"], value):
            self._code_options["co_consts"].append(value)
        idx = list_find_index_by_id(self._code_options["co_consts"], value)
        self._add_instr("LOAD_CONST", arg=idx, argval=value)

    def gen_print_log(self, message):
        """print a log :"""
        import paddle

        self.gen_load_object(
            paddle.fluid.core.set_eval_frame, "dbg_set_eval_frame"
        )
        self.gen_load_const(None)
        self.gen_call_function(1)
        self.gen_store_fast("old_eval_frame")
        self.gen_load_global("print")
        self.gen_load_const(message)
        self.gen_call_function(1)
        self.gen_pop_top()
        self.gen_load_object(
            paddle.fluid.core.set_eval_frame, "dbg_set_eval_frame"
        )
        self.gen_load_fast("old_eval_frame")
        self.gen_call_function(1)
        self.gen_pop_top()

    def gen_dbg_function(self, dbg_fun):
        """debug bytecode helper function.
        Usage like:
        def dbg_func():
            import inspect
            import dis
            print("dbg here.")
            print(locals())
            frame = inspect.currentframe().f_back
            code = (inspect.currentframe().f_back.f_code)
            breakpoint()
            print(inspect.currentframe().f_back.f_locals['y'])

        self.pycode_gen.gen_dbg_function(dbg_func)
        """
        import paddle

        self.gen_load_object(
            paddle.fluid.core.set_eval_frame, "dbg_set_eval_frame"
        )
        self.gen_load_const(None)
        self.gen_call_function(1)
        self.gen_store_fast("old_eval_frame")
        self.gen_load_object(dbg_fun, "dbg1")
        self.gen_call_function(0)
        self.gen_pop_top()
        self.gen_load_object(
            paddle.fluid.core.set_eval_frame, "dbg_set_eval_frame"
        )
        self.gen_load_fast("old_eval_frame")
        self.gen_call_function(1)
        self.gen_pop_top()

    def gen_load(self, name):
        if name in self._code_options["co_cellvars"]:
            self.gen_load_deref(name)
        elif name in self._code_options["co_varnames"]:
            self.gen_load_fast(name)
        elif name in self._code_options["co_names"]:
            self.gen_load_global(name)
        else:
            raise InnerError(
                f"Want gen_load, but {name} can not found in code object."
            )

    def gen_store(self, name, code):
        if name in code.co_cellvars:
            self.gen_store_deref(name)
        elif name in code.co_varnames:
            self.gen_store_fast(name)
        elif name in code.co_names:
            self.gen_store_global(name)
        else:
            raise InnerError(
                f"Want gen_store, but {name} can not found in code object."
            )

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

    def gen_delete_resume_locals(self):
        def dbg_func():
            import inspect

            print("dbg here.")
            frame = inspect.currentframe().f_back
            code = inspect.currentframe().f_back.f_code
            print("locals  = ", frame.f_locals)
            print("code is = ", code)
            import gc
            import sys

            gc.collect()
            if 'resume_0_stack_0' in frame.f_locals:
                print(
                    "Ref: ", sys.getrefcount(frame.f_locals['resume_0_stack_0'])
                )
                print(
                    "Refers: ",
                    gc.get_referrers(frame.f_locals['resume_0_stack_0']),
                )
                import sys

                sys.xk_args = frame.f_locals['resume_0_stack_0']

        # self.gen_dbg_function(dbg_func)
        resume_local_pattern = "resume_[0-9]+_stack_[0-9]+"
        for name in self._code_options['co_varnames']:
            if (
                re.match(resume_local_pattern, name)
                and name in self._frame.f_locals
            ):
                self.gen_delete_fast(name)

    def gen_delete_fast(self, name):
        if name not in self._code_options["co_varnames"]:
            self._code_options["co_varnames"].append(name)
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("DELETE_FAST", arg=idx, argval=name)

    def gen_load_deref(self, name):
        if name not in self._code_options["co_cellvars"]:
            self._code_options["co_cellvars"].append(name)
        idx = self._code_options["co_cellvars"].index(name)
        self._add_instr("LOAD_DEREF", arg=idx, argval=name)

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

    def gen_store_global(self, name):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("STORE_GLOBAL", arg=idx, argval=name)

    def gen_store_deref(self, name):
        if name not in self._code_options["co_cellvars"]:
            self._code_options["co_cellvars"].append(name)
        idx = self._code_options["co_cellvars"].index(name)
        self._add_instr("STORE_DEREF", arg=idx, argval=name)

    def gen_store_subscr(self):
        self._add_instr("STORE_SUBSCR")

    def gen_subscribe(self):
        self._add_instr("BINARY_SUBSCR")

    def gen_build_tuple(self, count):
        self._add_instr("BUILD_TUPLE", arg=count, argval=count)

    def gen_build_list(self, count):
        self._add_instr("BUILD_LIST", arg=count, argval=count)

    def gen_build_map(self, count):
        self._add_instr("BUILD_MAP", arg=count, argval=count)

    def gen_build_slice(self, argc):
        self._add_instr("BUILD_SLICE", arg=argc, argval=argc)

    def gen_unpack_sequence(self, count):
        self._add_instr("UNPACK_SEQUENCE", arg=count, argval=count)

    def gen_call_function(self, argc=0, with_eval_frame=False):
        if with_eval_frame:
            assert (
                self.disable_eval_frame
            ), "can only with eval frame when disable_eval_frame=True"
            self.gen_enable_eval_frame()
        self._add_instr("CALL_FUNCTION", arg=argc, argval=argc)
        if with_eval_frame:
            self.gen_disable_eval_frame()

    def gen_call_method(self, argc=0):
        self._add_instr("CALL_METHOD", arg=argc, argval=argc)

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
        if self.disable_eval_frame:
            self.gen_enable_eval_frame()
        # def dbg_fun():
        # pass
        # self.gen_dbg_function(dbg_fun)
        self._add_instr("RETURN_VALUE")

    def add_pure_instructions(self, instructions):
        """
        add instructions and do nothing.
        """
        if self.disable_eval_frame:
            self.gen_enable_eval_frame()
        self._instructions.extend(instructions)
        if self.disable_eval_frame:
            self.gen_disable_eval_frame()

    def _add_instr(self, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.append(instr)
        return instr

    def _insert_instr(self, index, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.insert(index, instr)

    def pprint(self):
        print('\n'.join(instrs_info(self._instructions)))

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
