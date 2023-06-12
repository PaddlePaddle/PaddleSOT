from .instruction_utils import (
    Instruction,
    calc_offset_from_bytecode_offset,
    convert_instruction,
    gen_instr,
    get_instructions,
    instrs_info,
    modify_extended_args,
    modify_instrs,
    modify_vars,
    relocate_jump_target,
    replace_instr,
    reset_offset,
)
from .opcode_analysis import analysis_inputs

__all__ = [
    "analysis_inputs",
    "calc_offset_from_bytecode_offset",
    "Instruction",
    "convert_instruction",
    "gen_instr",
    "get_instructions",
    "modify_instrs",
    "modify_vars",
    "reset_offset",
    "relocate_jump_target",
    "modify_extended_args",
    "replace_instr",
    "instrs_info",
]
