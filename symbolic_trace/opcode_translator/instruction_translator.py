import dataclasses
import dis
import opcode
from typing import Optional, Any
import sys, types

from .convert import convert_one, convert_multi, convert_return
from .opcode_info import *


TRACE_UTIL_NAMES = {
    "convert_one" : [-1, convert_one],
    "convert_multi": [-1, convert_multi],
    "convert_return": [-1, convert_return],
}


class InstructionTranslator:
    def __init__(self, frame, code_options):
        self.frame = frame
        self.instrs = list(map(convert_instruction, dis.get_instructions(frame.f_code)))
        self.jump_map = {}
        for instr in self.instrs:
            # for 3.8, see dis.py
            if instr.opcode in opcode.hasjrel:
                jump_offset = instr.offset + 2 + instr.arg
                instr.jump_to = self.instrs[jump_offset//2]
            elif instr.opcode in opcode.hasjabs:
                jump_offset = instr.arg
                instr.jump_to = self.instrs[jump_offset//2]

        self.code_options = code_options
        self.p = 0                  # a pointer

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

    def run(self):
        self.transform_opcodes_with_push()
        self.transform_return()
        modify_instrs(self)
        return self.instrs

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
        self.frame = instr_transformer.frame

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
        convert_multi_arg = TRACE_UTIL_NAMES["convert_multi"][0]
        instr = self.instr_trans.current_instr()
        instrs = [
            gen_instr("LOAD_GLOBAL", arg=convert_multi_arg, argval="convert_multi"),
            gen_instr("ROT_TWO"),
            gen_instr("CALL_FUNCTION", arg=1, argval=1),
            instr,
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

        if instr.opcode in ALL_JUMP:
            if instr.opcode in REL_JUMP:
                new_arg = instr.jump_to.offset - instr.offset - 2
            if instr.opcode in ABS_JUMP:
                new_arg = instr.jump_to.offset

            instr.arg = new_arg & 0xFF
            new_arg = new_arg >> 8
            for ex in reversed(extended_arg):
                ex.arg = new_arg & 0xFF
                new_arg = new_arg >> 8

            # need more extended_args instr
            # set arg in the first instruction
            if new_arg > 0:
                if extended_arg:
                    extended_arg[0].arg += new_arg << 8
                else:
                    instr.arg += new_arg << 8

        extended_arg.clear()

def modify_extended_args(instr_translator):
    modify_completed = True
    extend_args_record = {}
    for instr in instr_translator.instrs:
        if instr.arg and instr.arg >= 256:
            _instrs = [instr]
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
        modify_completed = False
        for key, val in extend_args_record.items():
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
    jump_to: Optional[int] = None
    is_generated: bool = True

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
