import paddle

from .opcode_translator import eval_frame_callback
from .utils import GraphLogger, log_do


def symbolic_translate(func):
    def impl(*args, **kwargs):
        paddle.fluid.core.set_eval_frame(eval_frame_callback)

        try:
            outs = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            paddle.fluid.core.set_eval_frame(None)

        log_do(4, lambda: print(GraphLogger()))

        return outs

    return impl
