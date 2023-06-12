from __future__ import annotations

from .instruction_utils import Instruction
from .opcode_info import ALL_JUMP, HAS_FREE, HAS_LOCAL


def analysis_inputs(
    instructions: list[Instruction],
    current_instr_idx: int,
    stop_instr_idx: int | None = None,
):
    writes = set()
    reads = set()
    visited = set()

    def walk(start):
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in visited:
                continue
            visited.add(i)

            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if (
                    instr.opname.startswith("LOAD")
                    and instr.argval not in writes
                ):
                    reads.add(instr.argval)
                elif instr.opname.startswith("STORE"):
                    writes.add(instr.argval)
            elif instr.opname in ALL_JUMP:
                assert instr.jump_to is not None
                target_idx = instructions.index(instr.jump_to)
                # Fork to two branches, jump or not
                walk(target_idx)

    walk(current_instr_idx)
    return reads
