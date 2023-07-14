import dis
from functools import partial

from ..utils import log, log_do
from .executor.opcode_executor import InstructionTranslatorCache
from .skip_files import need_skip


def print_locals(frame):
    local_key = [
        key for key in frame.f_locals.keys() if not key.startswith("__")
    ]
    print(
        f"[eval_frame_callback] {frame.f_code.co_name} with locals {local_key}"
    )
    print(
        f"[eval_frame_callback] {' ' * len(frame.f_code.co_name)} with cellvars + freevars:  {frame.f_code.co_cellvars + frame.f_code.co_freevars}"
    )

    def convert_obj(obj):
        import paddle

        if isinstance(obj, paddle.Tensor):
            return obj.shape
        if isinstance(obj, list):
            return [convert_obj(i) for i in obj]
        return obj

    for key in local_key:
        print(
            f"[eval_frame_callback] {' ' * len(frame.f_code.co_name)} {key} = {convert_obj(frame.f_locals[key])}"
        )


def eval_frame_callback(frame, **kwargs):
    # is generator
    if frame.f_code.co_flags & 0x20 > 0:
        return None

    if need_skip(frame.f_code):
        return None

    log(
        2,
        "[eval_frame_callback] start to translate: " + str(frame.f_code) + "\n",
    )
    log_do(4, partial(print_locals, frame))
    log(8, "[transform_opcode] old_opcode: " + frame.f_code.co_name + "\n")
    log_do(8, lambda: dis.dis(frame.f_code))

    new_code = InstructionTranslatorCache()(frame, **kwargs)

    log(
        7,
        "\n[transform_opcode] new_opcode:  " + frame.f_code.co_name + "\n",
    )
    if new_code is not None:
        log_do(7, lambda: dis.dis(new_code.code))
    else:
        log(7, f"Skip frame: {frame.f_code.co_name}")

    return new_code
