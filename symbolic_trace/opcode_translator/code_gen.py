from __future__ import annotations

import dataclasses
import dis
import sys
import types
from typing import Any, Optional


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


def gen_instr(name, arg=None, argval=None, gened=True):
    return Instruction(
        opcode=dis.opmap[name],
        opname=name,
        arg=arg,
        argval=argval,
        is_generated=gened,
    )
