from __future__ import annotations

import dataclasses
import dis
import opcode
from typing import Optional, Any
import sys, types
import functools

from .convert import convert_one, convert_multi, convert_return
from .opcode_info import *
from .opcode_generater import gen_new_opcode
from ..utils import freeze_structure, Singleton, Cache


TRACE_UTIL_NAMES = {
    "convert_one" : [-1, convert_one],
    "convert_multi": [-1, convert_multi],
    "convert_return": [-1, convert_return],
}

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

def locals_globals_injection(frame, code_options):
    # f_locals does not work
    global TRACE_UTIL_NAMES
    for key, val in TRACE_UTIL_NAMES.items():
        _, obj = val
        if key in frame.f_globals.keys() and not (frame.f_globals[key] is obj):
            raise(f"name {key} already exists!!!")
        if key in code_options["co_names"]:
            arg = code_options["co_names"].index(key)
            TRACE_UTIL_NAMES[key][0] = arg
            frame.f_globals[key] = obj
        else:
            arg = len(code_options["co_names"])
            TRACE_UTIL_NAMES[key][0] = arg
            code_options["co_names"].append(key)
            frame.f_globals[key] = obj

@functools.lru_cache(maxsize=128)
def gen_code_options(code):
    code_options = {}
    for k in pycode_attributes:
        val = getattr(code, k)
        if isinstance(val, tuple):
            val = list(val)
        code_options[k] = val
    return code_options

@Singleton
class InstructionTranslatorCache(Cache):
    def key_fn(self, *args, **kwargs):
        code, *others = args
        # TODO(@zhanfei): code optionals.
        return freeze_structure((code))

    def value_fn(self, *args, **kwargs):
        return InstructionTranslator.translate(*args, **kwargs)
        

class InstructionTranslator:
    def __init__(self):
        self.jump_map = {}
        self.p = 0                  # a pointer

    def gen_instructions(self, code):
        # instrs do not contain EXTENDED_ARG
        instrs = list(map(convert_instruction, dis.get_instructions(code)))
        for instr in instrs:
            # for 3.8, see dis.py
            if instr.opname in ALL_JUMP:
                if instr.opname in REL_JUMP:
                    origin_jump_target = instr.offset + 2 + instr.arg

                elif instr.opname in ABS_JUMP:
                    origin_jump_target = instr.arg

                jump_offset = origin_jump_target
                while instrs[jump_offset//2].opname == "EXTENDED_ARG":
                    jump_offset += 2

                if origin_jump_target != jump_offset:
                    # copy infos from EXETENDED_ARG to other opcode
                    if instrs[origin_jump_target//2].is_jump_target:
                        instrs[jump_offset//2].is_jump_target = instrs[origin_jump_target//2].is_jump_target
                    if instrs[origin_jump_target//2].starts_line:
                        instrs[jump_offset//2].starts_line = instrs[origin_jump_target//2].starts_line

                instr.jump_to = instrs[jump_offset//2]

        '''
        if the origin opcode contains EXTENDED_ARG, it should be like:
                
            >>  EXTENDED_ARG 1
                XX 388    <-  256 + 132

        filter all EXTENDED_ARG here
        '''
        instrs = [x for x in instrs if x.opname != "EXTENDED_ARG"]
        return instrs

    def current_instr(self):
        return self.instrs[self.p]

    def p_next(self, n=1):
        self.p += n

    def p_prev(self, n=1):
        self.p -= n

    def p_seek(self, n=0):
        self.p = n

    def p_find(self, instr):
        idx = self.instrs.index(instr)
        self.p_seek(idx)

    def find_next_instr(self, names):
        if isinstance(names, str):
            names = [names]
        found = False
        start = self.p + 1
        end = len(self.instrs)
        for i in range(start, end):
            if self.instrs[i].opname in names:
                found  = True
                self.p_seek(i)
                break
        return found

    def insert_instr(self, instr, idx=None):
        if idx is None:
            idx = self.p + 1
        self.instrs.insert(idx, instr)

    def remove_instr(self, idx=None):
        if idx is None:
            idx = self.p
        del self.instrs[idx]

    def replace_instr_list(self, instr_list, instr=None):
        if instr is None:
            part1 = self.instrs[0:self.p]
            part2 = self.instrs[self.p+1:]
            self.instrs = part1 + instr_list + part2
        else:
            self.p_find(instr)
            self.replace_instr_list(instr_list)

    def run(self, code):
        self.instrs = self.gen_instructions(code)
        self.transform_opcodes_with_push()
        self.transform_return()
        modify_instrs(self)
        return self.instrs

    @staticmethod
    def translate(code, code_options):
        self = InstructionTranslator()
        instrs = self.run(code)
        new_code = gen_new_opcode(instrs, code_options, pycode_attributes)
        return new_code

    def transform_opcodes_with_push(self):
        self.p_seek(-1)
        gener = InstrGen(self)

        while self.find_next_instr(ALL_WITH_PUSH):
            instr = self.current_instr()
            if instr.is_generated:
                continue

            if instr.opname in PUSH_ONE:
                to_be_replace = gener.gen_for_push_one()
                if to_be_replace:
                    self.replace_instr_list(to_be_replace)
                    self.p_next(len(to_be_replace)-1)
            elif instr.opname in PUSH_ARG:
                to_be_replace = gener.gen_for_push_arg()
                if to_be_replace:
                    self.replace_instr_list(to_be_replace)
                    self.p_next(len(to_be_replace)-1)

    def transform_return(self):
        self.p_seek(-1)
        gener = InstrGen(self)

        while self.find_next_instr(RETURN):
            instr = self.current_instr()
            if instr.is_generated:
                continue

            to_be_replace = gener.gen_for_return()
            if to_be_replace:
                self.replace_instr_list(to_be_replace)
                self.p_next(len(to_be_replace)-1)


class InstrGen:
    def __init__(self, instr_transformer):
        self.instr_trans = instr_transformer

    def gen_for_push_one(self):
        convert_one_arg = TRACE_UTIL_NAMES["convert_one"][0]
        instr = self.instr_trans.current_instr()
        instrs = [
            instr,
            gen_instr("LOAD_GLOBAL", arg=convert_one_arg, argval="convert_one"),
            gen_instr("ROT_TWO"),
            gen_instr("CALL_FUNCTION", arg=1, argval=1),
        ]
        return instrs
    
    def gen_for_push_arg(self):
        # the instrs do not contains EXETENDED_ARG, so we can directly transform the codes
        convert_multi_arg = TRACE_UTIL_NAMES["convert_multi"][0]
        instr = self.instr_trans.current_instr()
        instrs = [
            gen_instr("LOAD_GLOBAL", arg=convert_multi_arg, argval="convert_multi"),
            gen_instr("ROT_TWO"),
            gen_instr("CALL_FUNCTION", arg=1, argval=1),
            instr,          #  <-- no EXETENDED_ARG before instr
        ]
        return instrs
    
    def gen_for_return(self):
        convert_return_arg = TRACE_UTIL_NAMES["convert_return"][0]
        instr = self.instr_trans.current_instr()
        instrs = [
            gen_instr("LOAD_GLOBAL", arg=convert_return_arg, argval="convert_return"),
            gen_instr("ROT_TWO"),
            gen_instr("CALL_FUNCTION", arg=1, argval=1),
            instr,
        ]
        return instrs

def modify_instrs(instr_translator):
    modify_completed = False
    while not modify_completed:
        reset_offset(instr_translator)
        relocate_jump_target(instr_translator)
        modify_completed = modify_extended_args(instr_translator)

def reset_offset(instr_translator):
    for idx, instr in enumerate(instr_translator.instrs):
        instr.offset = idx * 2

def relocate_jump_target(instr_translator):
    extended_arg = []
    for instr in instr_translator.instrs:
        if instr.opname == "EXTENDED_ARG":
            extended_arg.append(instr)
            continue

        if instr.opname in ALL_JUMP:
            # if jump target has extended_arg, should jump to the first extended_arg opcode
            jump_target = instr.jump_to.offset if instr.jump_to.first_ex_arg is None else instr.jump_to.first_ex_arg.offset

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

def modify_extended_args(instr_translator):
    modify_completed = True
    extend_args_record = {}
    for instr in instr_translator.instrs:
        if instr.arg and instr.arg >= 256:      # more than one byte
            _instrs = [instr]                   # replace instr with _instrs later (it is a set of instrs), all operations will be recorded in extend_args_record
            val = instr.arg
            instr.arg = val & 0xFF
            val = val >> 8
            while val > 0:
                _instrs.append(
                    gen_instr("EXTENDED_ARG", arg=val & 0xFF)
                )
                val = val >> 8

            extend_args_record.update({instr : list(reversed(_instrs))})

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


@dataclasses.dataclass
class Instruction:
    opcode: int
    opname: str
    arg: Optional[int]
    argval: Any
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    jump_to: Optional[Instruction] = None
    is_generated: bool = True

    # for analys EXTENDED_ARG
    first_ex_arg: Optional[Instruction] = None
    ex_arg_for: Optional[Instruction] = None

    # used in modify_extended_args
    def __hash__(self):
        return id(self)


# convert dis.Instruction
def convert_instruction(instr):
    return Instruction(
        instr.opcode,
        instr.opname,
        instr.arg,
        instr.argval,
        instr.offset,
        instr.starts_line,
        instr.is_jump_target,
        jump_to=None,
        is_generated=False,
    )


def gen_instr(name, arg=None, argval=None, gened=True):
    return Instruction(
        opcode=dis.opmap[name], opname=name, arg=arg, argval=argval, is_generated=gened
    )


def instrs_info(instrs):
    ret = []
    for idx, instr in enumerate(instrs):
        if instr.starts_line is not None:
            ret.append("")
        ret.append("{line:<8s}{is_jump_target:>2s}{offset:>4d} {opname:<30s}{arg:<4s}{argval}".format(
            line=str(instr.starts_line) if instr.starts_line else "",
            is_jump_target=">>" if instr.is_jump_target else "  ",
            offset=idx * 2,
            opname=instr.opname,
            arg=str(instr.arg) if instr.arg else "",
            argval=f"({instr.argval})" if instr.argval else "",
        ))
    return "\n".join(ret)