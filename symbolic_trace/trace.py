import paddle

from .opcode_translator import eval_frame_callback
from .proxy_tensor import ProxyTensorContext


def symbolic_trace(func):
    def impl(*args, **kwargs):
        ProxyTensorContext().reset()
        paddle.fluid.core.set_eval_frame(eval_frame_callback)
        try:
            outs = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            paddle.fluid.core.set_eval_frame(None)
        return outs

    return impl
