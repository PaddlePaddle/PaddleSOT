# This class is used for abstract code generation:
# We only need to care about what type of bytecode our code needs to generate,
# without worrying about the subscripts of bytecode instructions in the code option.

from __future__ import annotations

import dis
import types

import opcode

from ..instruction_utils import gen_instr, modify_instrs, modify_vars

'''
    code options for PyCodeObject
'''

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


'''
    generator a new code object
'''


def gen_new_opcode(instrs, code_options, keys):
    bytecode, lnotab = assemble(instrs, code_options["co_firstlineno"])
    code_options["co_lnotab"] = lnotab
    code_options["co_code"] = bytecode
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    code_options["co_stacksize"] += 1
    for key, val in code_options.items():
        if isinstance(val, list):
            code_options[key] = tuple(val)
    # code_options is a dict, use keys to makesure the input order
    return types.CodeType(*[code_options[k] for k in keys])


# list of instructions => bytecode & lnotab
def assemble(instructions, firstlineno):
    cur_line = firstlineno
    cur_bytecode = 0

    code = []
    lnotab = []

    for instr in instructions:
        # set lnotab
        if instr.starts_line is not None:
            line_offset = instr.starts_line - cur_line
            bytecode_offset = len(code) - cur_bytecode

            cur_line = instr.starts_line
            cur_bytecode = len(code)

            lnotab.extend(modify_lnotab(bytecode_offset, line_offset))

        # get bytecode
        arg = instr.arg or 0
        code.extend((instr.opcode, arg & 0xFF))

    return bytes(code), bytes(lnotab)


def to_byte(num):
    if num < 0:
        num += 256  #  -1 => 255
    return num


def modify_lnotab(byte_offset, line_offset):
    if byte_offset > 127:
        ret = []
        while byte_offset > 127:
            ret.extend((127, 0))
            byte_offset -= 127
        # line_offset might > 127, call recursively
        ret.extend(modify_lnotab(byte_offset, line_offset))
        return ret

    if line_offset > 127:
        # here byte_offset < 127
        ret = [byte_offset, 127]
        line_offset -= 127
        while line_offset > 0:
            ret.extend((0, line_offset))
            line_offset -= 127
        return ret

    # both < 127
    return [to_byte(byte_offset), to_byte(line_offset)]


# TODO: need to update
def stacksize(instructions):
    cur_stack = 0
    max_stacksize = 0

    for instr in instructions:
        if instr.opcode in opcode.hasjabs or instr.opcode in opcode.hasjrel:
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=True)
        else:
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=False)
        cur_stack += stack_effect
        assert cur_stack >= 0
        if cur_stack > max_stacksize:
            max_stacksize = cur_stack
    return max_stacksize


'''
    helper to create new code object
'''


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
        modify_vars(self._instructions, self._code_options)
        new_code = gen_new_opcode(
            self._instructions, self._code_options, pycode_attributes
        )
        return new_code

    def gen_load_const(self, value):
        if value in self._code_options["co_consts"]:
            idx = self._code_options["co_consts"].index(value)
        else:
            idx = len(self._code_options["co_consts"])
            self._code_options["co_consts"].append(value)
        self._add_instr("LOAD_CONST", arg=idx, argval=value)

    def gen_load_object(self, obj, obj_name):
        if obj_name not in self.objname_map:
            self._f_globals[obj_name] = obj
            self._code_options["co_names"].append(obj_name)
            idx = len(self._code_options["co_names"]) - 1
            self.objname_map[obj_name] = idx
        idx = self.objname_map[obj_name]
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=obj_name)

    def gen_store_fast(self, name):
        if name not in self._code_options["co_varnames"]:
            self._code_options["co_varnames"].append(name)
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("STORE_FAST", arg=idx, argval=name)

    def gen_load_fast(self, name):
        assert name in self._code_options["co_varnames"]
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("LOAD_FAST", arg=idx, argval=name)

    def gen_build_tuple(self, count):
        self._add_instr("BUILD_TUPLE", arg=count, argval=count)

    def gen_build_list(self, count):
        self._add_instr("BUILD_LIST", arg=count, argval=count)

    def gen_build_map(self, count):
        self._add_instr("BUILD_MAP", arg=count, argval=count)

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

    def _insert_instr(self, index, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.insert(index, instr)

    def pprint(self):
        for instr in self._instructions:
            print(instr.opname, "\t\t", instr.argval)
