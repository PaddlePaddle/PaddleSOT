import contextlib
import paddle
from .opcode_translator import eval_frame_callback
from .symbolic_trace import SymbolicTraceContext
from .proxy_tensor import ProxyTensorContext, ProxyTensor
from .statement_ir import Symbol

def symbolic_trace(func):
    def wrapped(*args, **kw):
        ProxyTensorContext().reset()
        with SymbolicTraceContext() as ctx:
            paddle.fluid.core.set_eval_frame(eval_frame_callback)
            returns = func(*args, **kw)
            paddle.fluid.core.set_eval_frame(None)

        # TODO( output analysis, we can get out symbols here. )
        ret = SymbolicTraceContext().start_return(
            ProxyTensorContext().get_runtime(),
            output=returns)
        return ret
    return wrapped
