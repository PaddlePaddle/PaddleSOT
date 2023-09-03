from .instruction_utils import (
    Instruction,
    calc_offset_from_bytecode_offset,
    calc_stack_effect,
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
from .opcode_analysis import (
    Space,
    analysis_inputs,
    analysis_used_names_with_space,
)

__all__ = [
    "analysis_inputs",
    "analysis_used_names_with_space",
    "calc_offset_from_bytecode_offset",
    "calc_stack_effect",
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
    "Space",
]
