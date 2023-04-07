from __future__ import annotations

# TODO: Refactor this file

import dis
import inspect

HASLOCAL_OPCODES = set(dis.haslocal)
HASFREE_OPCODES = set(dis.hasfree)
COMPARE_OPCODES = set(dis.cmp_op)
HASJREL_OPCODES = set(dis.hasjrel)
HASJABS_OPCODES = set(dis.hasjabs)
JUMP_OPCODES = HASJREL_OPCODES | HASJABS_OPCODES
import types

def calc_offset_from_bytecode_offset(bytecode_offset: int) -> int:
    # Calculate the index from bytecode offset, because it have 2 bytes per instruction
    # TODO: Change this for Python 3.11+.
    return bytecode_offset // 2


def calc_jump_target(
    instructions: list[dis.Instruction], current_instr_idx: int
) -> int:
    """
    Handle the case where the jump target is in the middle of an extended arg.
    """
    num_instr = len(instructions)
    # For each opcode, at most three prefixal EXTENDED_ARG are allowed, so we
    # need to check at most 4 instructions.
    # See more details in https://docs.python.org/3.10/library/dis.html#opcode-EXTENDED_ARG 
    for i in range(current_instr_idx, min(current_instr_idx + 4, num_instr)):
        if instructions[i].opcode != dis.EXTENDED_ARG:
            return i
    else:
        raise ValueError("Could not find jump target")


def read_write_analysis(instructions: list[dis.Instruction], current_instr_idx: int):
    writes = set()
    reads = set()
    visited = set()

    def walk(start):
        for i in range(start, len(instructions)):
            if i in visited:
                continue
            visited.add(i)

            instr = instructions[i]
            if instr.opcode in HASLOCAL_OPCODES | HASFREE_OPCODES:
                if instr.opname.startswith("LOAD") and instr.argval not in writes:
                    reads.add(instr.argval)
                elif instr.opname.startswith("STORE"):
                    writes.add(instr.argval)
            elif instr.opcode in JUMP_OPCODES:
                target_idx = calc_offset_from_bytecode_offset(instr.argval)
                target_idx = calc_jump_target(instructions, target_idx)
                # Fork to two branches, jump or not
                walk(target_idx)

    walk(current_instr_idx)
    return reads


def output_analysis(frame: types.FrameType):
    instructions = list(dis.get_instructions(frame.f_code))

    current_instr_idx = calc_offset_from_bytecode_offset(frame.f_lasti)
    reads = read_write_analysis(instructions, current_instr_idx)
    return reads