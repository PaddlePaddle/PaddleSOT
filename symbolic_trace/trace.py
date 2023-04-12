import contextlib
import paddle
from .opcode_translator import ConvertGuard, eval_frame_callback
from .symbolic.symbolic_context import SymbolicTraceContext
from .proxy_tensor import ProxyTensorContext, ProxyTensor
from .convert_functions import convert_function

def symbolic_trace(func):
    def symbolic_traced_func(*args, **kw):
        ProxyTensorContext().reset()
        with SymbolicTraceContext() as ctx:
            with ConvertGuard(convert_function) as ctx:
                paddle.fluid.core.set_eval_frame(eval_frame_callback)
                try:
                    returns = func(*args, **kw)
                except Exception as e:
                    raise e
                finally: 
                    paddle.fluid.core.set_eval_frame(None)
        ret = SymbolicTraceContext().start_compile(
            ProxyTensorContext(),
            output=returns)
        return ret
    return symbolic_traced_func
