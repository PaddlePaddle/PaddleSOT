import dis
import collections
from .instruction_translator import InstructionTranslatorCache, convert_instruction, pycode_attributes, locals_globals_injection, gen_code_options
from .opcode_generater import gen_new_opcode
from .skip_translate_names import SKIP_TRANSLATE_NAMES
from ..utils import log_do, log, no_eval_frame
from .convert import Callbacks
import paddle
import functools

CustomCode = collections.namedtuple("CustomCode", ["code"])

class ConvertGuard:
    def __init__(self, on_convert=None):
        self.on_convert = on_convert
        
    def __enter__(self):
        log(1, "[start_bytecode_transform] start transform\n")
        assert self.on_convert is None or callable(self.on_convert)
        self.old = Callbacks().set_on_convert(self.on_convert)
    
    def __exit__(self, exc_type, exc_value, traceback):
        log(1, "[start_bytecode_transform] end transform\n")
        Callbacks().set_on_convert(self.old)

def eval_frame_callback(frame):
    if frame.f_code.co_name not in SKIP_TRANSLATE_NAMES:
        log(2, "[eval_frame_callback] want translate: " + frame.f_code.co_name + "\n")
        new_code = transform_opcode(frame)
        retval = CustomCode(new_code)
        return retval
    return None

def transform_opcode(frame):
    # check definition in types.CodeType

    code_options = gen_code_options(frame.f_code)
    locals_globals_injection(frame, code_options)
    new_code = InstructionTranslatorCache()(
        frame.f_code, code_options
    )
    log(4, "[transform_opcode] old_opcode: " + frame.f_code.co_name + "\n")
    log_do(4, lambda: dis.dis(frame.f_code))


    log(7, "\n[transform_opcode] new_opcode:  " + frame.f_code.co_name + "\n")
    log_do(7, lambda: dis.dis(new_code))

    return new_code

def print_instrs(instrs):
    for instr in instrs:
        if instr.starts_line is not None:
            print("{:<4d}{}".format(instr.starts_line, instr.opname))
        else:
            print("    {}".format(instr.opname))
