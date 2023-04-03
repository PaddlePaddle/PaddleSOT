import contextlib
import paddle

from .opcode_translator import eval_frame_callback, enable_log
from .symbolic_trace import SymbolicTraceContext
from .proxy_tensor import clear_runtime_proxytensor

def symbolic_trace(func, with_log=True):
    def wrapped(*args, **kw):
        with SymbolicTraceGuard(eval_frame_callback):
            func(*args, **kw)
    return wrapped

@contextlib.contextmanager
def SymbolicTraceGuard(callback):
    with SymbolicTraceContext() as ctx:
        clear_runtime_proxytensor()
        paddle.fluid.core.set_eval_frame(callback)
        yield
        paddle.fluid.core.set_eval_frame(None)
        SymbolicTraceContext().start_compile()

