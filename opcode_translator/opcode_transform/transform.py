import dis
import collections
from .instruction_translator import InstructionTranslator, convert_instruction
from .opcode_generater import gen_new_opcode
from .convert import CONVERT_SKIP_NAMES

CustomCode = collections.namedtuple("CustomCode", ["code"])

def eval_frame_callback(frame):

    if frame.f_code.co_name not in CONVERT_SKIP_NAMES:
        new_code = transform_opcode(frame)
        retval = CustomCode(new_code)
        return retval
    return None


def transform_opcode(frame):
    # check definition in types.CodeType
    keys = [
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_filename",
        "co_name",
        "co_firstlineno",
        "co_lnotab",
        "co_freevars",
        "co_cellvars",
    ]

    code_options = {}
    for k in keys:
        val = getattr(frame.f_code, k)
        if isinstance(val, tuple):
            val = list(val)
        code_options[k] = val

    instr_gen = InstructionTranslator(frame, code_options)
    instrs = instr_gen.run()
    new_code = gen_new_opcode(instrs, code_options, keys, frame)
    #dis.dis(frame.f_code)
    dis.dis(new_code)
    return new_code

def print_instrs(instrs):
    for instr in instrs:
        if instr.starts_line is not None:
            print("{:<4d}{}".format(instr.starts_line, instr.opname))
        else:
            print("    {}".format(instr.opname))
