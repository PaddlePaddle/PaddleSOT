# This class is used for abstract code generation:
# We only need to care about what type of bytecode our code needs to generate,
# without worrying about the subscripts of bytecode instructions in the code option.

from __future__ import annotations

import dis
import types

import opcode

from ..instruction_utils import gen_instr, modify_instrs


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
    def __init__(self, f_globals, f_code):
        self._origin_code = f_code
        self._code_options = gen_code_options(self._origin_code)
        self._f_globals = f_globals
        self._instructions = []
        # map from name to LOAD_GLOBAL/LOAD_ATTR/STORE_GLOBAL/STORE_ATTR index
        self.co_names_argval2arg : Dict[str, int] = {}
        # map from varname to LOAD_FAST/STORE_FAST index
        self.co_varnames_argval2arg : Dict[str, int] = {}
        # map from const to LOAD_CONST index
        self.co_consts_argval2arg : Dict[str, int] = {}

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
        return self.load_global(obj, obj_name)

    def load_global(self, obj, obj_name):
        idx, inserted = self._get_name_arg_and_inserted(argval=obj_name)
        if inserted:
            self._f_globals[obj_name] = obj
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=obj_name)

    def store_global(self, name):
        name_index = self._get_name_arg(name)
        self._add_instr("STORE_GLOBAL", arg=name_index, argval=name)

    def load_attr(self, attr_name):
        name_index = self._get_name_arg(attr_name)
        self._add_instr("LOAD_ATTR", arg=name_index, argval=attr_name)

    def import_name(self, name):
        name_index = self._get_name_arg(name)
        self._add_instr("IMPORT_NAME", arg=name_index, argval=name)

    def load_method(self, method_name):
        name_index = self._get_name_arg(method_name)
        self._add_instr("LOAD_METHOD", arg=name_index, argval=method_name)

    def load_const(self, obj):
        name_index = self._get_const_arg(obj)
        self._add_instr("LOAD_CONST", arg=name_index, argval=obj)

    def load_fast(self, varname):
        name_index = self._get_varname_arg(varname)
        self._add_instr("LOAD_FAST", arg=name_index, argval=varname)

    def store_fast(self, varname):
        name_index = self._get_varname_arg(varname)
        self._add_instr("STORE_FAST", arg=name_index, argval=varname)

    def gen_call_function(self, argc=0):
        self.call_function(argc=argc)

    def call_function(self, argc=0):
        self._add_instr("CALL_FUNCTION", arg=argc, argval=argc)

    def call_method(self, argc=0):
        self._add_instr("CALL_METHOD", arg=argc, argval=argc)

    def pop_top(self):
        self._add_instr("POP_TOP", arg=None, argval=None)

    def gen_return(self):
        self._add_instr("RETURN_VALUE")

    def return_value(self):
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

    def pprint(self):
        for instr in self._instructions:
            print(instr.opname, "\t\t", instr.argval)

    def _get_name_arg(self, argval):
        return self._get_name_arg_and_inserted(argval)[0]

    def _get_name_arg_and_inserted(self, argval):
        return self._get_arg_and_inserted(
            arg_map_name="co_names",
            argval2arg=self.co_names_argval2arg,
            argval=argval
        )

    def _get_varname_arg(self, argval):
        return self._get_varname_arg_and_inserted(argval)[0]

    def _get_varname_arg_and_inserted(self, argval):
        return self._get_arg_and_inserted(
            arg_map_name="co_varnames",
            argval2arg=self.co_varnames_argval2arg,
            argval=argval
        )

    def _get_const_arg(self, argval):
        return self._get_const_arg_and_inserted(argval)[0]

    def _get_const_arg_and_inserted(self, argval):
        return self._get_arg_and_inserted(
            arg_map_name="co_consts",
            argval2arg=self.co_consts_argval2arg,
            argval=argval
        )

    def _get_arg_and_inserted(self, arg_map_name, argval2arg, argval):
        if argval not in argval2arg:
            self._code_options[arg_map_name].append(argval)
            idx = len(self._code_options[arg_map_name]) - 1
            argval2arg[argval] = idx
            return argval2arg[argval], True
        else:
            return argval2arg[argval], False
