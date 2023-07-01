from __future__ import annotations

import dataclasses

from .instruction_utils import Instruction
from .opcode_info import ALL_JUMP, HAS_FREE, HAS_LOCAL, UNCONDITIONAL_JUMP


@dataclasses.dataclass
class State:
    reads: set[str]
    writes: set[str]
    visited: set[int]


def analysis_inputs(
    instructions: list[Instruction],
    current_instr_idx: int,
    stop_instr_idx: int | None = None,
) -> set[str]:
    """
    Create a state object with empty sets for Analyzes the instructions.
    Iterate through the instructions starting from the current index until the stop index,check the instructions and update state object set.

    Args:
        instructions (list): List of Instruction objects representing bytecode instructions.
        current_instr_idx (int): The index of the current instruction being analyzed.
        stop_instr_idx (int, optional): The stopping index for the analysis. If None, analyze until the end. Default is None.

    Returns:
        set: A set of strings representing the inputs used by the instructions.
    """
    root_state = State(set(), set(), set())

    def fork(
        state: State, start: int, jump: bool, jump_target: int
    ) -> set[str]:
        new_start = start + 1 if not jump else jump_target
        new_state = State(
            set(state.reads), set(state.writes), set(state.visited)
        )
        return walk(new_state, new_start)

    def walk(state: State, start: int) -> set[str]:
        """
        Performs the analysis of bytecode instructions recursively starting from a given start index.
        """
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in state.visited:
                return state.reads
            state.visited.add(i)

            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if instr.opname.startswith("LOAD") and instr.argval not in (
                    state.writes
                ):
                    state.reads.add(instr.argval)
                elif instr.opname.startswith("STORE"):
                    state.writes.add(instr.argval)
            elif instr.opname in ALL_JUMP:
                assert instr.jump_to is not None
                target_idx = instructions.index(instr.jump_to)
                # Fork to two branches, jump or not
                jump_branch = fork(state, i, True, target_idx)
                not_jump_branch = (
                    fork(state, i, False, target_idx)
                    if instr.opname not in UNCONDITIONAL_JUMP
                    else set()
                )
                return jump_branch | not_jump_branch
            elif instr.opname == "RETURN_VALUE":
                return state.reads
        return state.reads

    return walk(root_state, current_instr_idx)
