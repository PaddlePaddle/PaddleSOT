from __future__ import annotations

import dataclasses
import sys

from .instruction_utils import Instruction
from .opcode_info import ALL_JUMP, HAS_FREE, HAS_LOCAL

sys.path.append("/home/pfcc/PaddleSOT/")


@dataclasses.dataclass
class ReadsWrites:
    reads: set
    writes: set
    visited: set


def analysis_inputs(
    instructions: list[Instruction],
    current_instr_idx: int,
    stop_instr_idx: int | None = None,
):
    must = ReadsWrites(set(), set(), set())
    may = ReadsWrites(set(), set(), set())

    def walk(state, start):
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in state.visited:
                continue
            state.visited.add(i)

            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if (
                    instr.opname.startswith("LOAD")
                    and instr.argval not in state.writes
                ):
                    state.reads.add(instr.argval)
                elif instr.opname.startswith("STORE"):
                    state.writes.add(instr.argval)
            elif instr.opname in ALL_JUMP:
                assert instr.jump_to is not None
                target_idx = instructions.index(instr.jump_to)
                # Fork to two branches, jump or not
                walk(may, target_idx)

    walk(must, current_instr_idx)
    return must.reads | may.reads
