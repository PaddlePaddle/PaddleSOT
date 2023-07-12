from __future__ import annotations

import dataclasses

from .instruction_utils import Instruction
from .opcode_info import ALL_JUMP, HAS_FREE, HAS_LOCAL, UNCONDITIONAL_JUMP


@dataclasses.dataclass
class State:
    reads: set[str]
    writes: set[str]
    visited: set[int]


def is_read_opcode(opname):
    if opname in ["LOAD_FAST", "LOAD_DEREF", "LOAD_NAME", "LOAD_GLOBAL"]:
        return True
    if opname in (
        "DELETE_FAST",
        "DELETE_DEREF",
        "DELETE_NAME",
        "DELETE_GLOBAL",
    ):
        return True
    return False


def is_write_opcode(opname):
    if opname in ["STORE_FAST", "STORE_NAME", "STORE_DEREF", "STORE_GLOBAL"]:
        return True
    if opname in (
        "DELETE_FAST",
        "DELETE_DEREF",
        "DELETE_NAME",
        "DELETE_GLOBAL",
    ):
        return True
    return False


def analysis_inputs(
    instructions: list[Instruction],
    current_instr_idx: int,
    stop_instr_idx: int | None = None,
):
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
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in state.visited:
                return state.reads
            state.visited.add(i)

            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if is_read_opcode(instr.opname) and instr.argval not in (
                    state.writes
                ):
                    state.reads.add(instr.argval)
                elif is_write_opcode(instr.opname):
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


def analysis_inputs_outputs(
    instructions: list[Instruction],
    current_instr_idx: int,
    stop_instr_idx: int | None = None,
):
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
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in state.visited:
                return state.reads | state.writes
            state.visited.add(i)

            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if is_read_opcode(instr.opname) and instr.argval not in (
                    state.writes
                ):
                    state.reads.add(instr.argval)
                elif is_write_opcode(instr.opname):
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
                return state.reads | state.writes
        return state.reads | state.writes

    return walk(root_state, current_instr_idx)
