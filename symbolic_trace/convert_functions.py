from .utils import no_eval_frame, log
from .proxy_tensor import ProxyTensorContext, callable_wrapper
import paddle

log_level = 10

def convert_callable(func):
    if isinstance(func, type):
        return func
    return callable_wrapper(func)

def convert_tensor(tensor):
    return ProxyTensorContext().from_tensor(tensor)

@no_eval_frame
def convert_function(obj):
    if callable(obj):
        log(log_level, "found a callable object\n")
        obj = convert_callable(obj)
    elif isinstance(obj, paddle.Tensor):
        log(log_level, "found a tensor\n")
        obj = convert_tensor(obj)
    else:
        log(log_level, "nothing happend\n")
    return obj
