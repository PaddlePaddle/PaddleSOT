import contextlib
import paddle
from .opcode_translator import eval_frame_callback
from .symbolic_trace import SymbolicTraceContext
from .proxy_tensor import ProxyTensorContext

def symbolic_trace(func, with_log=False):
    def wrapped(*args, **kw):
        ProxyTensorContext().reset()
        with SymbolicTraceContext():
            paddle.fluid.core.set_eval_frame(eval_frame_callback)
            func(*args, **kw)
            paddle.fluid.core.set_eval_frame(None)
        return SymbolicTraceContext().start_compile(ProxyTensorContext().get_runtime())
    return wrapped
