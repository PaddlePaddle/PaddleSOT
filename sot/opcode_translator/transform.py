from __future__ import annotations

import dis
from functools import partial
from typing import TYPE_CHECKING

from ..utils import CodeStatus, EventGuard, log, log_do
from .executor.opcode_executor import CustomCode, InstructionTranslatorCache
from .skip_files import need_skip

if TYPE_CHECKING:
    pass


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
            return "Tensor(" + str(obj.shape) + ")"
        if isinstance(obj, list):
            return [convert_obj(i) for i in obj]
        return obj

    for key in local_key:
        print(
            f"[eval_frame_callback] {' ' * len(frame.f_code.co_name)} {key} = {convert_obj(frame.f_locals[key])}"
        )


def eval_frame_callback(frame, **kwargs):
    with EventGuard(
        f"eval_frame_callback: {frame.f_code.co_name}", event_level=2
    ):
        # is generator
        if frame.f_code.co_flags & 0x20 > 0:
            return None

        if need_skip(frame):
            return None

        log(
            2,
            "[eval_frame_callback] start to translate: "
            + str(frame.f_code)
            + "\n",
        )
        log_do(4, partial(print_locals, frame))

        log(3, f"[transform] OriginCode: {frame.f_code.co_name}\n")
        log_do(3, lambda: dis.dis(frame.f_code))

        custom_code = InstructionTranslatorCache()(frame, **kwargs)

        if (
            custom_code is None
            or custom_code.code == frame.f_code
            and custom_code.disable_eval_frame is False
            or custom_code.code is None
        ):
            log(
                3,
                "[transform] NewCode (same as origin code): "
                + frame.f_code.co_name
                + "\n",
            )
            log_do(3, lambda: dis.dis(frame.f_code))
            return None
        else:
            if CodeStatus().check_code(custom_code.code):
                log(3, "[transform] Code has found no graph, block it.")
                return CustomCode(frame.f_code, True)
            log(
                3,
                "[transform] NewCode: " + custom_code.code.co_name + "\n",
            )
            log_do(3, lambda: dis.dis(custom_code.code))
            return custom_code
