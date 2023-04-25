import paddle

from .opcode_translator import eval_frame_callback
from .proxy_tensor import ProxyTensorContext
from .symbolic.symbolic_context import SymbolicTraceContext


def symbolic_trace(func):
    def symbolic_traced_func(*args, **kwargs):
        ProxyTensorContext().reset()
        with SymbolicTraceContext():
            paddle.fluid.core.set_eval_frame(eval_frame_callback)
            try:
                returns = func(*args, **kwargs)
            except Exception as e:
                raise e
            finally:
                paddle.fluid.core.set_eval_frame(None)
        return returns

    return symbolic_traced_func
