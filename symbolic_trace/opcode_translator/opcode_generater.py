import dis
import opcode
import sys, types
import dataclasses
from numbers import Real


def gen_new_opcode(instrs, code_options, keys, frame):
    bytecode, lnotab = assemble(instrs, code_options["co_firstlineno"])
    code_options["co_lnotab"] = lnotab
    code_options["co_code"] = bytecode
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    code_options["co_stacksize"] = stacksize(instrs)
    for key, val in code_options.items():
        if isinstance(val, list):
            code_options[key] = tuple(val)
    # code_options is a dict, use keys to makesure the input order
    return types.CodeType(*[code_options[k] for k in keys])


def assemble(instructions, firstlineno):
    cur_line = firstlineno
    cur_bytecode = 0

    code = []
    lnotab = []

    for instr in instructions:
        # set lnotab
        if instr.starts_line is not None:
            line_offset = max(-128, min(instr.starts_line - cur_line, 127))
            bytecode_offset_offset = max(0, min(len(code) - cur_bytecode, 255))
            assert line_offset != 0 or bytecode_offset_offset != 0
            cur_line = instr.starts_line
            cur_bytecode = len(code)
            lnotab.extend((bytecode_offset_offset, line_offset))

        # get bytecode
        arg = instr.arg or 0
        code.extend((instr.opcode, arg & 0xFF))
    return bytes(code), bytes(lnotab)


def stacksize(instructions):
    cur_stack = 0
    max_stacksize = 0

    for instr in instructions:
        if instr.opcode in opcode.hasjabs or instr.opcode in opcode.hasjrel:
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=True)
        else:
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=False)
        cur_stack += stack_effect
        assert cur_stack >= 0
        if cur_stack > max_stacksize:
            max_stacksize = cur_stack
    return max_stacksize
