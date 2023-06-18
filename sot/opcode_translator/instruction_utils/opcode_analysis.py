from __future__ import annotations

import dataclasses

from .instruction_utils import Instruction
from .opcode_info import ALL_JUMP, HAS_FREE, HAS_LOCAL


@dataclasses.dataclass(frozen=True)
class State:
    reads: set[str]
    writes: set[str]
    visited: set[int]

    def __hash__(self):
        return hash(
            (
                frozenset(self.reads),
                frozenset(self.writes),
                frozenset(self.visited),
            )
        )


def analysis_inputs(
    instructions: list[Instruction],
    current_instr_idx: int,
    stop_instr_idx: int | None = None,
):
    root_state = State(set(), set(), set())
    branches: dict[tuple[int, bool, State], set[str]] = {
        # (current_instr_idx, True, root_state): root_state
    }

    def fork(
        state: State, start: int, jump: bool, jump_target: int
    ) -> set[str]:
        # The cache key contains: start of jump, whether jump or not, original state
        cache_key = (start, jump, state)
        new_start = start + 1 if not jump else jump_target
        new_state = State(
            set(state.reads), set(state.writes), set(state.visited)
        )
        if cache_key not in branches:
            reads = walk(new_state, new_start)
            branches[cache_key] = reads
            return reads
        return branches[cache_key]

    def walk(state: State, start: int) -> set[str]:
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in state.visited:
                continue
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
                not_jump_branch = fork(state, i, False, target_idx)
                jump_branch = fork(state, i, True, target_idx)
                return not_jump_branch | jump_branch
            elif instr.opname == "RETURN_VALUE":
                return state.reads
        return set()

    return walk(root_state, current_instr_idx)
