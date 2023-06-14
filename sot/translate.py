import paddle

from .opcode_translator import eval_frame_callback
from .opcode_translator.executor.opcode_executor import OpcodeExecutorBase


def symbolic_translate(func):
    def impl(*args, **kwargs):
        paddle.fluid.core.set_eval_frame(eval_frame_callback)
        try:
            outs = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            paddle.fluid.core.set_eval_frame(None)
            OpcodeExecutorBase.call_stack = []
        return outs

    return impl
