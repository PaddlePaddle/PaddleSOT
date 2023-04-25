# This class is used for abstract code generation:
# We only need to care about what type of bytecode our code needs to generate,
# without worrying about the subscripts of bytecode instructions in the code option.

from __future__ import annotations

import dataclasses
import dis
import functools
import sys
import types
from typing import Any, Optional

import opcode

from ..utils import (
    Cache,
    InnerError,
    Singleton,
    UnsupportError,
    freeze_structure,
    log,
)
from .code_gen import Instruction, gen_instr
from .opcode_generater import gen_new_opcode
from .opcode_info import *

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
    "co_lnotab",
    "co_freevars",
    "co_cellvars",
]


def gen_code_options(code):
    code_options = {}
    for k in pycode_attributes:
        val = getattr(code, k)
        if isinstance(val, tuple):
            val = list(val)
        code_options[k] = val
    return code_options


def modify_extended_args(instructions):
    modify_completed = True
    extend_args_record = {}
    for instr in instructions:
        if instr.arg and instr.arg >= 256:  # more than one byte
            _instrs = [
                instr
            ]  # replace instr with _instrs later (it is a set of instrs), all operations will be recorded in extend_args_record
            val = instr.arg
            instr.arg = val & 0xFF
            val = val >> 8
            while val > 0:
                _instrs.append(gen_instr("EXTENDED_ARG", arg=val & 0xFF))
                val = val >> 8

            extend_args_record.update({instr: list(reversed(_instrs))})

    if extend_args_record:
        # if new EXTENDED_ARG inserted, we need update offset and jump target
        modify_completed = False

        def bind_ex_arg_with_instr(ex_arg, instr):
            # move opcode info to EXTENDED_ARG
            ex_arg.starts_line = instr.starts_line
            instr.starts_line = None
            ex_arg.is_jump_target = instr.is_jump_target
            instr.is_jump_target = False

            if instr.ex_arg_for is not None:
                # instr is also an ex_arg for another instr
                instr.ex_arg_for.first_ex_arg = ex_arg
                ex_arg.ex_arg_for = instr.ex_arg_for
                instr.ex_arg_for = None
            else:
                instr.first_ex_arg = ex_arg
                ex_arg.ex_arg_for = instr

        for key, val in extend_args_record.items():
            bind_ex_arg_with_instr(val[0], key)
            instr_translator.replace_instr_list(val, key)

    return modify_completed


def relocate_jump_target(instuctions):
    extended_arg = []
    for instr in instuctions:
        if instr.opname == "EXTENDED_ARG":
            extended_arg.append(instr)
            continue

        if instr.opname in ALL_JUMP:
            # if jump target has extended_arg, should jump to the first extended_arg opcode
            jump_target = (
                instr.jump_to.offset
                if instr.jump_to.first_ex_arg is None
                else instr.jump_to.first_ex_arg.offset
            )

            if instr.opname in REL_JUMP:
                new_arg = jump_target - instr.offset - 2
            elif instr.opname in ABS_JUMP:
                new_arg = jump_target

            if extended_arg:
                instr.arg = new_arg & 0xFF
                new_arg = new_arg >> 8
                for ex in reversed(extended_arg):
                    ex.arg = new_arg & 0xFF
                    new_arg = new_arg >> 8

                # need more extended_args instr
                # set arg in the first extended_arg
                if new_arg > 0:
                    extended_arg[0].arg += new_arg << 8
            else:
                instr.arg = new_arg

        extended_arg.clear()


def reset_offset(instructions):
    for idx, instr in enumerate(instructions):
        instr.offset = idx * 2


def modify_instrs(instructions):
    modify_completed = False
    while not modify_completed:
        reset_offset(instructions)
        relocate_jump_target(instructions)
        modify_completed = modify_extended_args(instructions)


class PyCodeGen:
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
        new_code = gen_new_opcode(
            self._instructions, self._code_options, pycode_attributes
        )
        return new_code

    def gen_load_object(self, obj, obj_name):
        if obj_name not in self.objname_map:
            self._f_globals[obj_name] = obj
            self._code_options["co_names"].append(obj_name)
            idx = len(self._code_options["co_names"]) - 1
            self.objname_map[obj_name] = idx
        idx = self.objname_map[obj_name]
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=obj_name)

    def gen_call_function(self, argc=0):
        self._add_instr("CALL_FUNCTION", arg=argc, argval=argc)

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

    def pprint(self):
        for instr in self._instructions:
            print(instr.opname, "\t\t", instr.argval)
