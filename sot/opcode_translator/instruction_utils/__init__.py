from .instruction_utils import (  # noqa: F401
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
from .opcode_analysis import (  # noqa: F401
    Space,
    analysis_inputs,
    analysis_used_names_with_space,
)
