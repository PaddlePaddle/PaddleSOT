import contextlib
import paddle
from .opcode_translator import ConvertGuard, eval_frame_callback
from .symbolic_context import SymbolicTraceContext
from .proxy_tensor import ProxyTensorContext, ProxyTensor
from .statement_ir import Symbol
from .convert_functions import convert_function

def symbolic_trace(func):
    def wrapped(*args, **kw):
        ProxyTensorContext().reset()
        with SymbolicTraceContext() as ctx:
            with ConvertGuard(convert_function) as ctx:
                paddle.fluid.core.set_eval_frame(eval_frame_callback)
                returns = func(*args, **kw)
                paddle.fluid.core.set_eval_frame(None)
        # TODO( output analysis, we can get out symbols here. )
        ret = SymbolicTraceContext().start_return(
            ProxyTensorContext().get_runtime(),
            output=returns)
        return ret
    return wrapped
