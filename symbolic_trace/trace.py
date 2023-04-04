import contextlib
import paddle
from .opcode_translator import eval_frame_callback
from .symbolic_trace import SymbolicTraceContext
from .proxy_tensor import ProxyTensorContext

def symbolic_trace(func, with_log=False):
    def wrapped(*args, **kw):
        with SymbolicTraceGuard(eval_frame_callback):
            func(*args, **kw)
        return SymbolicTraceContext().start_compile(ProxyTensorContext().get_runtime())
    return wrapped

@contextlib.contextmanager
def SymbolicTraceGuard(callback):
    with SymbolicTraceContext() as ctx:
        ProxyTensorContext().reset()
        paddle.fluid.core.set_eval_frame(callback)
        yield
        paddle.fluid.core.set_eval_frame(None)

