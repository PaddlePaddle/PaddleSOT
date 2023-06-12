import dis

from ..utils import log, log_do
from .executor.opcode_executor import InstructionTranslatorCache
from .skip_files import need_skip_path


def eval_frame_callback(frame):
    # is generator
    if frame.f_code.co_flags & 0x20 > 0:
        return None

    if not need_skip_path(frame.f_code.co_filename):
        log(
            2,
            "[eval_frame_callback] start to translate: "
            + frame.f_code.co_name
            + "\n",
        )

        log(8, "[transform_opcode] old_opcode: " + frame.f_code.co_name + "\n")
        log_do(8, lambda: dis.dis(frame.f_code))

        new_code = InstructionTranslatorCache()(frame)

        log(
            7,
            "\n[transform_opcode] new_opcode:  " + frame.f_code.co_name + "\n",
        )
        if new_code is not None:
            log_do(7, lambda: dis.dis(new_code.code))
        else:
            log_do(7, f"Skip frame: {frame.f_code.co_name}")

        return new_code
    return None
