import collections
import dis

from ..utils import log, log_do

from .executor import InstructionTranslatorCache
from .skip_files import need_skip_path

CustomCode = collections.namedtuple("CustomCode", ["code"])


def eval_frame_callback(frame):
    if not need_skip_path(frame.f_code.co_filename):
        log(
            2,
            "[eval_frame_callback] want translate: "
            + frame.f_code.co_name
            + "\n",
        )

        log(8, "[transform_opcode] old_opcode: " + frame.f_code.co_name + "\n")
        log_do(8, lambda: dis.dis(frame.f_code))

        new_code = InstructionTranslatorCache()(frame)

        log(7, "\n[transform_opcode] new_opcode:  " + frame.f_code.co_name + "\n")
        log_do(7, lambda: dis.dis(new_code))

        retval = CustomCode(new_code)
        return retval
    return None