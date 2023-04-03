import dataclasses
import dis
import opcode
from typing import Optional, Any
import sys, types

from .convert import convert_one, convert_multi
from .opcode_info import *


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


ADD_GLOBAL_NAMES = {
    "convert_one" : [-1, convert_one],
    "convert_multi": [-1, convert_multi],
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
        global ADD_GLOBAL_NAMES
        for key, val in ADD_GLOBAL_NAMES.items():
            _, obj = val
            if key in frame.f_globals.keys() and not (frame.f_globals[key] is obj):
                raise(f"name {key} already exists!!!")
            if key in code_options["co_names"]:
                arg = code_options["co_names"].index(key)
                ADD_GLOBAL_NAMES[key][0] = arg
                frame.f_globals[key] = obj
            else:
                arg = len(code_options["co_names"])
                ADD_GLOBAL_NAMES[key][0] = arg
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

    def replace_instr_list(self, instr_list):
        part1 = self.instrs[0:self.p]
        part2 = self.instrs[self.p+1:]
        self.instrs = part1 + instr_list + part2

    def run(self):
        self.transform_opcodes_with_push()
        self.modify_instrs()
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

    def modify_instrs(self):
        # TODO: consider extend_args
        for idx, instr in enumerate(self.instrs):
            instr.offset = idx * 2
        
        for instr in self.instrs:
            if instr.opcode in opcode.hasjrel:
                instr.arg = instr.jump_to.offset - instr.offset - 2
            elif instr.opcode in opcode.hasjabs:
                instr.arg = instr.jump_to.offset

class InstrGen:
    def __init__(self, instr_transformer):
        self.instr_trans = instr_transformer
        self.frame = instr_transformer.frame

    def gen_for_push_one(self):
        convert_one_arg = ADD_GLOBAL_NAMES["convert_one"][0]
        instr = self.instr_trans.current_instr()
        instrs = [
            instr,
            gen_instr("LOAD_GLOBAL", arg=convert_one_arg, argval="convert_one"),
            gen_instr("ROT_TWO"),
            gen_instr("CALL_FUNCTION", arg=1, argval=1),
        ]
        return instrs
    
    def gen_for_push_arg(self):
        convert_multi_arg = ADD_GLOBAL_NAMES["convert_multi"][0]
        instr = self.instr_trans.current_instr()
        instrs = [
            gen_instr("LOAD_GLOBAL", arg=convert_multi_arg, argval="convert_multi"),
            gen_instr("ROT_TWO"),
            gen_instr("CALL_FUNCTION", arg=1, argval=1),
            instr,
        ]
        return instrs

